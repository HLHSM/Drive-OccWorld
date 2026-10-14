"""Static batch-one BEVFormer graph, preserving TSA and all AMoE experts."""
import hashlib
import copy
import numpy as np
import torch
from torch import nn
from deploy.cuda_ops import attention

class ImageEncoder(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.backbone, self.neck = model.img_backbone, model.img_neck
    def forward(self, images):
        return tuple(self.neck(self.backbone(images)))

class GroupNormExport(nn.Module):
    """Explicit reductions: works with TensorRT 8.4 and 5D volumes."""
    def __init__(self, norm):
        super().__init__()
        self.groups, self.eps = norm.num_groups, norm.eps
        self.weight, self.bias = norm.weight, norm.bias
    def forward(self, x):
        grouped = x.reshape(x.shape[0], self.groups, -1)
        centered = grouped - grouped.mean(-1, keepdim=True)
        normalized = centered / torch.sqrt(centered.square().mean(-1, keepdim=True)+self.eps)
        normalized = normalized.reshape_as(x)
        shape = [1, x.shape[1]] + [1]*(x.ndim-2)
        return normalized*self.weight.reshape(shape)+self.bias.reshape(shape)

def export_norms(module):
    for name, child in list(module.named_children()):
        if isinstance(child, nn.GroupNorm):
            setattr(module, name, GroupNormExport(child))
        else:
            export_norms(child)

class OccupancyHead(nn.Module):
    def __init__(self, model, for_export=False):
        super().__init__()
        head = model.future_pred_head
        self.lift, self.amoe, self.out = head.agri_amoe_lift, copy.deepcopy(head.agri_amoe), head.agri_amoe_out
        self.height, self.width, self.depth = head.bev_h, head.bev_w, head.num_pred_height
        self.channels = head.agri_amoe_channels
        if for_export:
            export_norms(self.amoe)
    def forward(self, bev):
        volume = self.lift(bev).reshape(1, self.height, self.width, self.channels,
                                       self.depth).permute(0,3,4,1,2).contiguous()
        output = self.out(self.amoe(volume))
        return output.permute(0,3,4,2,1).reshape(1,self.height*self.width,self.depth,-1)

class BEVEncoder(nn.Module):
    def __init__(self, model, metadata, features, backend='cuda'):
        super().__init__()
        head = model.pts_bbox_head
        transformer = head.transformer
        encoder = transformer.encoder
        if getattr(encoder, 'use_nearfar_bev', False):
            raise ValueError('This deployment preserves the dense base TSA encoder; NearFar is not enabled')
        self.layers = encoder.layers
        for layer in self.layers:
            if type(layer.attentions[0]).__name__ != 'TemporalSelfAttention':
                raise ValueError('Expected original TSA encoder, not GVAD or another mixer')
        self.height, self.width = head.bev_h, head.bev_w
        self.backend = backend
        self.query = head.bev_embedding
        mask = features[0].new_zeros((1,self.height,self.width))
        position = head.positional_encoding(mask).flatten(2).transpose(1,2)
        self.register_buffer('position', position.detach())
        self.register_buffer('cam_embed', transformer.cams_embeds[:features[0].shape[0]].detach())
        self.register_buffer('level_embed', transformer.level_embeds.detach())
        self.use_cams_embeds = transformer.use_cams_embeds
        shapes = torch.tensor([[f.shape[-2],f.shape[-1]] for f in features],device=mask.device,dtype=torch.int32)
        self.register_buffer('shapes',shapes)
        self.register_buffer('starts',torch.cat([shapes.new_zeros(1),shapes.prod(1).cumsum(0)[:-1]]).int())
        self.register_buffer('normalizer',shapes.flip(-1).float())
        self.register_buffer('bev_shapes',shapes.new_tensor([[self.height,self.width]]))
        self.register_buffer('bev_starts',shapes.new_zeros(1))
        reference = encoder.get_reference_points(self.height,self.width,dim='2d',bs=1,device=mask.device,dtype=torch.float32)
        self.register_buffer('reference_2d',reference.repeat(2,1,1,1))
        reference3 = encoder.get_reference_points(self.height,self.width,encoder.pc_range[5]-encoder.pc_range[2],
                    encoder.num_points_in_pillar,dim='3d',bs=1,device=mask.device,dtype=torch.float32)
        projected,visible = encoder.point_sampling(reference3,encoder.pc_range,[metadata])
        all_indices=[v[0].sum(-1).nonzero().flatten() for v in visible]
        self.camera_count=len(all_indices)
        self.length=max(map(len,all_indices))
        reference_camera=projected.new_zeros(self.camera_count,self.length,projected.shape[3],2)
        gather=torch.zeros(self.camera_count,self.length,dtype=torch.long,device=mask.device)
        inverse=torch.full((self.camera_count,self.height*self.width),self.length,dtype=torch.long,device=mask.device)
        for camera,index in enumerate(all_indices):
            reference_camera[camera,:len(index)]=projected[camera,0,index]
            gather[camera,:len(index)]=index
            inverse[camera,index]=torch.arange(len(index),device=mask.device)
        self.register_buffer('reference_camera',reference_camera)
        self.register_buffer('gather',gather)
        self.register_buffer('inverse',inverse)
        self.register_buffer('count',visible.sum(-1).gt(0).permute(1,2,0).sum(-1).clamp(min=1).float().unsqueeze(-1))
        self.geometry_hash=geometry_hash(metadata,self.height,self.width)

    def temporal(self, module, query):
        # Keep both distinct learned sampling groups. Only the duplicated
        # value projection is shared, rather than deleting the TSA pathway.
        residual=query
        positioned=query+self.position
        source=torch.cat([query,positioned],-1)
        heads,points=module.num_heads,module.num_points
        value=module.value_proj(query).repeat(2,1,1).reshape(2,-1,heads,module.embed_dims//heads)
        offsets=module.sampling_offsets(source).reshape(1,-1,heads,2,1,points,2)
        weights=module.attention_weights(source).reshape(1,-1,heads,2,points).softmax(-1)
        weights=weights.permute(0,3,1,2,4).reshape(2,-1,heads,1,points)
        offsets=offsets.permute(0,3,1,2,4,5,6).reshape(2,-1,heads,1,points,2)
        locations=self.reference_2d[:,:,None,:,None,:]+offsets/offsets.new_tensor([self.width,self.height])
        output=attention(value,self.bev_shapes,self.bev_starts,locations,weights,self.backend)
        output=output.reshape(2,-1,module.embed_dims).mean(0,keepdim=True)
        return residual+module.output_proj(output)

    def spatial(self, module, query, value):
        deform=module.deformable_attention
        cameras,length,channels=value.shape
        heads,levels,points=deform.num_heads,deform.num_levels,deform.num_points
        rebatch=query[0,self.gather]
        offsets=deform.sampling_offsets(rebatch).reshape(cameras,self.length,heads,levels,points,2)
        weights=deform.attention_weights(rebatch).reshape(cameras,self.length,heads,levels*points).softmax(-1)
        weights=weights.reshape(cameras,self.length,heads,levels,points)
        anchors=self.reference_camera.shape[2]
        offsets=offsets.reshape(cameras,self.length,heads,levels,points//anchors,anchors,2)
        locations=self.reference_camera[:,:,None,None,None,:,:]+offsets/self.normalizer[None,None,None,:,None,None,:]
        locations=locations.reshape(cameras,self.length,heads,levels,points,2)
        projected=deform.value_proj(value).reshape(cameras,length,heads,channels//heads)
        output=attention(projected,self.shapes,self.starts,locations,weights,self.backend)
        # Static gather replaces nonzero/rebatch/scatter and is TRT 8.4 native.
        output=torch.cat([output,output.new_zeros(cameras,1,channels)],1)
        slots=output[0,self.inverse[0]]
        for camera in range(1,self.camera_count):
            slots=slots+output[camera,self.inverse[camera]]
        return query+module.output_proj(slots.unsqueeze(0)/self.count)

    def forward(self, *features):
        values=[]
        for level,feature in enumerate(features):
            flat=feature.flatten(2).transpose(1,2)
            if self.use_cams_embeds:
                flat=flat+self.cam_embed[:,None,:]
            values.append(flat+self.level_embed[level:level+1,None,:])
        value=torch.cat(values,1)
        query=self.query.weight.unsqueeze(0)
        for layer in self.layers:
            query=layer.norms[0](self.temporal(layer.attentions[0],query))
            query=layer.norms[1](self.spatial(layer.attentions[1],query,value))
            query=layer.norms[2](layer.ffns[0](query))
        return query

def geometry_hash(metadata,height,width):
    payload=np.asarray(metadata['lidar2img'],dtype=np.float64).tobytes()
    payload+=str((metadata['img_shape'],height,width)).encode()
    return hashlib.sha256(payload).hexdigest()
