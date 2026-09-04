import copy
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F

from mmdet.models import HEADS, build_loss

from mmcv.runner import force_fp32, auto_fp16
from .world_head_base import WorldHeadBase
from projects.mmdet3d_plugin.bevformer.losses.semkitti_loss import geo_scal_loss, sem_scal_loss, CE_ssc_loss
from projects.mmdet3d_plugin.bevformer.losses.lovasz_softmax import lovasz_softmax


class _AnisotropicDepthwise3DBlock(nn.Module):
    """Low-cost XY/Z factorized residual block for FarmSim occupancy."""

    def __init__(self, channels):
        super().__init__()
        groups = 8 if channels % 8 == 0 else 1
        self.xy = nn.Conv3d(channels, channels, (1, 3, 3), padding=(0, 1, 1),
                            groups=channels, bias=False)
        self.z = nn.Conv3d(channels, channels, (3, 1, 1), padding=(1, 0, 0),
                           groups=channels, bias=False)
        self.norm = nn.GroupNorm(groups, channels)
        self.pointwise = nn.Conv3d(channels, channels, 1, bias=False)
        self.out_norm = nn.GroupNorm(groups, channels)

    def forward(self, feature):
        residual = feature
        feature = F.relu(self.norm(self.xy(feature)), inplace=True)
        feature = F.relu(self.z(feature), inplace=True)
        feature = self.out_norm(self.pointwise(feature))
        return F.relu(feature + residual, inplace=True)


@HEADS.register_module()
class WorldHeadV1(WorldHeadBase):
    def __init__(self,
                 history_queue_length,
                 soft_weight,
                 loss_weight_cfg=None,
                 output_scale=2,
                 use_crop_gap_refinement=False,
                 crop_gap_crop_class=1,
                 crop_gap_boundary_loss_weight=0.5,
                 crop_gap_free_loss_weight=0.25,
                 crop_gap_alpha=3.0,
                 crop_gap_sigma=1.5,
                 crop_gap_radius=4,
                 use_selective_c2f=False,
                 c2f_active_ratio=0.25,
                 c2f_channels=128,
                 # ADHR is training-only agricultural dual-hardness mining.
                 # It is independent of the removed uncertainty-refinement path.
                 use_dual_hardness_refinement=False,
                 dual_hardness_active_ratio=0.04,
                 dual_hardness_gap_ratio=0.5,
                 dual_hardness_channels=128,
                 dual_hardness_local_scale=0.25,
                 dual_hardness_gap_boost=0.5,
                 dual_hardness_loss_weight=0.5,
                 dual_hardness_distill_weight=0.1,
                 dual_hardness_ema_decay=0.99,
                 use_gap_residual_refiner=False,
                 gap_refiner_channels=24,
                 gap_refiner_blocks=3,
                 gap_refiner_coarse_loss_weight=0.15,
                 gap_refiner_boundary_loss_weight=0.25,
                 gap_refiner_gap_loss_weight=0.5,
                 gap_refiner_crop_loss_weight=0.5,
                 gap_refiner_use_bev_feature=True,
                 gap_refiner_use_image_features=False,
                 gap_refiner_image_active_ratio=0.08,
                 gap_refiner_image_channels=24,
                 gap_refiner_image_levels=2,
                 gap_refiner_image_crop_ratio=0.5,
                 *args,
                 **kwargs):
        super().__init__(*args, **kwargs)

        self.history_queue_length = history_queue_length    # 2
        self.output_scale=output_scale
        self.use_crop_gap_refinement = bool(use_crop_gap_refinement)
        self.crop_gap_crop_class = int(crop_gap_crop_class)
        self.crop_gap_boundary_loss_weight = float(crop_gap_boundary_loss_weight)
        self.crop_gap_free_loss_weight = float(crop_gap_free_loss_weight)
        self.crop_gap_alpha = float(crop_gap_alpha)
        self.crop_gap_sigma = float(crop_gap_sigma)
        self.crop_gap_radius = int(crop_gap_radius)
        self._crop_gap_boundary_logits = None
        self.use_selective_c2f = bool(use_selective_c2f)
        self.c2f_active_ratio = float(c2f_active_ratio)
        self.c2f_channels = int(c2f_channels)
        self.use_dual_hardness_refinement = bool(use_dual_hardness_refinement)
        self.dual_hardness_active_ratio = float(dual_hardness_active_ratio)
        self.dual_hardness_gap_ratio = float(dual_hardness_gap_ratio)
        self.dual_hardness_channels = int(dual_hardness_channels)
        self.dual_hardness_local_scale = float(dual_hardness_local_scale)
        self.dual_hardness_gap_boost = float(dual_hardness_gap_boost)
        self.dual_hardness_loss_weight = float(dual_hardness_loss_weight)
        self.dual_hardness_distill_weight = float(dual_hardness_distill_weight)
        self.dual_hardness_ema_decay = float(dual_hardness_ema_decay)
        self._dual_hardness_features = None
        # Dense but lightweight BEV-conditioned 3D residual refinement.  Its
        # optional image branch samples FPN only at selected difficult/crop
        # cells, then feeds that evidence back into this same refinement head.
        self.use_gap_residual_refiner = bool(use_gap_residual_refiner)
        self.gap_refiner_channels = int(gap_refiner_channels)
        self.gap_refiner_blocks = int(gap_refiner_blocks)
        self.gap_refiner_coarse_loss_weight = float(
            gap_refiner_coarse_loss_weight)
        self.gap_refiner_boundary_loss_weight = float(
            gap_refiner_boundary_loss_weight)
        self.gap_refiner_gap_loss_weight = float(gap_refiner_gap_loss_weight)
        self.gap_refiner_crop_loss_weight = float(gap_refiner_crop_loss_weight)
        self.gap_refiner_use_bev_feature = bool(gap_refiner_use_bev_feature)
        self.gap_refiner_use_image_features = bool(
            gap_refiner_use_image_features)
        self.gap_refiner_image_active_ratio = float(
            gap_refiner_image_active_ratio)
        self.gap_refiner_image_channels = int(gap_refiner_image_channels)
        self.gap_refiner_image_levels = int(gap_refiner_image_levels)
        self.gap_refiner_image_crop_ratio = float(
            gap_refiner_image_crop_ratio)
        self._gap_refiner_coarse_logits = None
        self._gap_refiner_gate_logits = None
        if self.crop_gap_boundary_loss_weight < 0:
            raise ValueError('crop_gap_boundary_loss_weight must be non-negative.')
        if self.crop_gap_free_loss_weight < 0:
            raise ValueError('crop_gap_free_loss_weight must be non-negative.')
        if self.crop_gap_alpha < 0:
            raise ValueError('crop_gap_alpha must be non-negative.')
        if self.crop_gap_sigma <= 0:
            raise ValueError('crop_gap_sigma must be positive.')
        if self.crop_gap_radius < 1:
            raise ValueError('crop_gap_radius must be at least 1.')
        if not 0 <= self.crop_gap_crop_class < self.num_classes:
            raise ValueError('crop_gap_crop_class must be a valid semantic class.')
        if not 0.0 < self.c2f_active_ratio <= 1.0:
            raise ValueError('c2f_active_ratio must be in (0, 1].')
        if self.c2f_channels < 8:
            raise ValueError('c2f_channels must be at least 8.')
        if not 0.0 < self.dual_hardness_active_ratio <= 1.0:
            raise ValueError('dual_hardness_active_ratio must be in (0, 1].')
        if not 0.0 <= self.dual_hardness_gap_ratio <= 1.0:
            raise ValueError('dual_hardness_gap_ratio must be in [0, 1].')
        if self.dual_hardness_channels < 8:
            raise ValueError('dual_hardness_channels must be at least 8.')
        if min(self.dual_hardness_local_scale, self.dual_hardness_gap_boost,
               self.dual_hardness_loss_weight,
               self.dual_hardness_distill_weight) < 0:
            raise ValueError('ADHR loss weights must be non-negative.')
        if not 0.0 <= self.dual_hardness_ema_decay < 1.0:
            raise ValueError('dual_hardness_ema_decay must be in [0, 1).')
        if self.gap_refiner_channels < 8:
            raise ValueError('gap_refiner_channels must be at least 8.')
        if self.gap_refiner_blocks < 1:
            raise ValueError('gap_refiner_blocks must be at least 1.')
        if min(self.gap_refiner_coarse_loss_weight,
               self.gap_refiner_boundary_loss_weight,
               self.gap_refiner_gap_loss_weight,
               self.gap_refiner_crop_loss_weight) < 0:
            raise ValueError('gap-refiner loss weights must be non-negative.')
        if not 0.0 < self.gap_refiner_image_active_ratio <= 1.0:
            raise ValueError('gap_refiner_image_active_ratio must be in (0, 1].')
        if self.gap_refiner_image_channels < 8:
            raise ValueError('gap_refiner_image_channels must be at least 8.')
        if self.gap_refiner_image_levels < 1:
            raise ValueError('gap_refiner_image_levels must be at least 1.')
        if not 0.0 <= self.gap_refiner_image_crop_ratio <= 1.0:
            raise ValueError('gap_refiner_image_crop_ratio must be in [0, 1].')
        if self.use_crop_gap_refinement and soft_weight:
            raise ValueError(
                'Crop-gap refinement requires the direct 2D occupancy decoder.')
        if self.use_selective_c2f and soft_weight:
            raise ValueError(
                'Selective C2F refinement requires the direct 2D occupancy decoder.')
        if self.use_dual_hardness_refinement and soft_weight:
            raise ValueError(
                'ADHR requires the direct 2D occupancy decoder.')
        if self.use_gap_residual_refiner and soft_weight:
            raise ValueError(
                'Gap residual refiner requires the direct 2D occupancy decoder.')

        self.class_weights = np.ones((self.num_classes,))
        self.class_weights[1:] = 5
        self.class_weights = torch.from_numpy(self.class_weights)

        # voxel sem losses
        if loss_weight_cfg is None:
            self.multi_loss = False
            self.loss_voxel_ce_weight = 1.0
        else:
            self.multi_loss = True
            self.loss_voxel_ce_weight = loss_weight_cfg.get('loss_voxel_ce_weight', 1.0)
            self.loss_voxel_sem_scal_weight = loss_weight_cfg.get('loss_voxel_sem_scal_weight', 1.0)
            self.loss_voxel_lovasz_weight = loss_weight_cfg.get('loss_voxel_lovasz_weight', 1.0)
            self.loss_voxel_geo_scal_weight = loss_weight_cfg.get('loss_voxel_geo_scal_weight', 1.0)

        self.soft_weight = soft_weight
        self._init_bev_pred_layers()

        if self.use_crop_gap_refinement:
            # The branch predicts a 3D crop/free boundary confidence from the
            # final BEV decoder feature and uses it only to gate a residual.
            # Zero initialization preserves the dense baseline at iteration 0.
            self.crop_gap_boundary_head = nn.Linear(
                self.embed_dims, self.num_pred_height)
            self.crop_gap_residual = nn.Sequential(
                nn.Linear(self.embed_dims, self.embed_dims // 2),
                nn.LayerNorm(self.embed_dims // 2),
                nn.ReLU(inplace=True),
                nn.Linear(self.embed_dims // 2,
                          self.num_pred_height * self.num_classes))
            nn.init.zeros_(self.crop_gap_residual[-1].weight)
            nn.init.zeros_(self.crop_gap_residual[-1].bias)

        if self.use_selective_c2f:
            # Four learned sub-query offsets represent the 2x2 child cells of
            # a selected coarse BEV query. Only the highest crop/free
            # uncertainty cells are decoded, avoiding a dense 200x200 head.
            self.c2f_subquery_offsets = nn.Parameter(
                torch.empty(4, self.embed_dims))
            nn.init.normal_(self.c2f_subquery_offsets, std=0.02)
            self.c2f_subquery_decoder = nn.Sequential(
                nn.Linear(self.embed_dims, self.c2f_channels),
                nn.LayerNorm(self.c2f_channels),
                nn.ReLU(inplace=True),
                nn.Linear(self.c2f_channels,
                          self.num_pred_height * self.num_classes))
            nn.init.zeros_(self.c2f_subquery_decoder[-1].weight)
            nn.init.zeros_(self.c2f_subquery_decoder[-1].bias)

        if self.use_dual_hardness_refinement:
            # HASSC-inspired, training-only voxel refinement. A crop/free
            # quota is combined with uncertainty sampling in the loss path,
            # so the public inference graph stays unchanged.
            self.dual_hardness_z_embed = nn.Embedding(
                self.num_pred_height, self.embed_dims)
            self.dual_hardness_refiner = nn.Sequential(
                nn.Linear(self.embed_dims, self.dual_hardness_channels),
                nn.LayerNorm(self.dual_hardness_channels),
                nn.ReLU(inplace=True),
                nn.Linear(self.dual_hardness_channels, self.num_classes))
            nn.init.zeros_(self.dual_hardness_refiner[-1].weight)
            nn.init.zeros_(self.dual_hardness_refiner[-1].bias)
            self.dual_hardness_teacher_z_embed = copy.deepcopy(
                self.dual_hardness_z_embed)
            self.dual_hardness_teacher = copy.deepcopy(
                self.dual_hardness_refiner)
            for parameter in self.dual_hardness_teacher_z_embed.parameters():
                parameter.requires_grad_(False)
            for parameter in self.dual_hardness_teacher.parameters():
                parameter.requires_grad_(False)

        if self.use_gap_residual_refiner:
            # The BEV feature still contains image evidence that the coarse
            # semantic MLP may have smoothed away.  Compress it before the
            # 3D operation and use anisotropic depthwise kernels because the
            # targeted error is predominantly crop/free structure in XY.
            refiner_input_channels = self.num_classes + 2
            if self.gap_refiner_use_bev_feature:
                self.gap_refiner_bev_proj = nn.Conv2d(
                    self.embed_dims, self.gap_refiner_channels, 1)
                refiner_input_channels += self.gap_refiner_channels
            self.gap_refiner_stem = nn.Sequential(
                nn.Conv3d(refiner_input_channels, self.gap_refiner_channels,
                          1, bias=False),
                nn.GroupNorm(
                    8 if self.gap_refiner_channels % 8 == 0 else 1,
                    self.gap_refiner_channels),
                nn.ReLU(inplace=True))
            self.gap_refiner_blocks_module = nn.Sequential(*[
                _AnisotropicDepthwise3DBlock(self.gap_refiner_channels)
                for _ in range(self.gap_refiner_blocks)])
            self.gap_refiner_gate = nn.Conv3d(
                self.gap_refiner_channels, 1, 1)
            self.gap_refiner_delta = nn.Conv3d(
                self.gap_refiner_channels, self.num_classes, 1)
            # The baseline prediction is exactly preserved at initialization.
            nn.init.zeros_(self.gap_refiner_delta.weight)
            nn.init.zeros_(self.gap_refiner_delta.bias)
            if self.gap_refiner_use_image_features:
                # Sparse current-image evidence: ambiguous crop/free cells and
                # confident crop cells are projected to FPN, fused across views,
                # then injected into the dense 3D GapRef branch.  A zero final
                # projection preserves the original GapRef at initialization.
                self.gap_refiner_image_proj = nn.Linear(
                    self.embed_dims, self.gap_refiner_image_channels)
                self.gap_refiner_image_view_attn = nn.Sequential(
                    nn.Linear(self.embed_dims + self.gap_refiner_image_channels,
                              self.gap_refiner_image_channels),
                    nn.ReLU(inplace=True),
                    nn.Linear(self.gap_refiner_image_channels, 1))
                self.gap_refiner_image_fuse = nn.Sequential(
                    nn.Conv3d(self.gap_refiner_image_channels + 1,
                              self.gap_refiner_channels, 1),
                    nn.GroupNorm(
                        8 if self.gap_refiner_channels % 8 == 0 else 1,
                        self.gap_refiner_channels),
                    nn.ReLU(inplace=True),
                    nn.Conv3d(self.gap_refiner_channels,
                              self.gap_refiner_channels, 1))
                nn.init.zeros_(self.gap_refiner_image_fuse[-1].weight)
                nn.init.zeros_(self.gap_refiner_image_fuse[-1].bias)

        self.num_points_sampling_feat = self.transformer.decoder.num_layers
        if self.soft_weight:
            self.bev_soft_weights = nn.Sequential(
                nn.Linear(self.embed_dims//2, self.embed_dims//2),
                nn.LayerNorm(self.embed_dims//2),
                nn.ReLU(inplace=True),
                nn.Linear(self.embed_dims//2, self.num_points_sampling_feat),
            )

            self.occ_pred_conv = nn.Sequential(
                nn.Linear(self.embed_dims//2, self.embed_dims//2),
                nn.LayerNorm(self.embed_dims//2),
                nn.ReLU(inplace=True),
                nn.Linear(self.embed_dims//2, self.num_pred_height * self.num_classes)
            )

    def _init_bev_pred_layers(self):
        """Overwrite the {self.bev_pred_head} of super()._init_layers()
        """
        bev_pred_branch = []
        mid_dims = self.embed_dims//2 if self.soft_weight else self.embed_dims
        for _ in range(self.num_pred_fcs):
            bev_pred_branch.append(nn.Linear(self.embed_dims, mid_dims))
            bev_pred_branch.append(nn.LayerNorm(mid_dims))
            bev_pred_branch.append(nn.ReLU(inplace=True))

        # not_soft_weight: direct output
        if not self.soft_weight:
            bev_pred_branch.append(nn.Linear(
                mid_dims, self.num_pred_height * self.num_classes))

        bev_pred_head = nn.Sequential(*bev_pred_branch)

        def _get_clones(module, N):
            return nn.ModuleList([copy.deepcopy(module) for i in range(N)])

        # Auxiliary supervision for all intermediate results.
        num_pred = self.transformer.decoder.num_layers if self.transformer.decoder.return_intermediate else 1
        self.bev_pred_head = _get_clones(bev_pred_head, num_pred)

    def forward_head_soft(self, next_bev_feats):
        """Get freespace estimation from multi-frame BEV feature maps.

        Args:
            next_bev_feats (torch.Tensor): with shape as
                [pred_frame_num_set, inter_num, bs, bev_h * bev_w, dims]    pred_frame_num_set = cur + future_select
        """
        next_bev_preds = []
        for lvl in range(next_bev_feats.shape[1]):
            next_bev_preds.append(self.bev_pred_head[lvl](next_bev_feats[:, lvl]))
        
        if self.soft_weight:
            bev_soft_weights = self.bev_soft_weights(next_bev_preds[-1])
            bev_soft_weights = torch.softmax(bev_soft_weights, dim=1)
        else:
            bev_soft_weights = torch.ones([next_bev_preds[-1].shape[0], next_bev_preds[-1].shape[1], 1, self.num_points_sampling_feat], ).to(next_bev_preds[0].device) / self.num_points_sampling_feat
        
        # soft_weight
        out_bev_feats = 0
        for feat, weights in zip(next_bev_preds, torch.unbind(bev_soft_weights, dim=-1)):
            out_bev_feats += feat * weights.unsqueeze(-1)
        
        # out pred
        out_occ = self.occ_pred_conv(out_bev_feats) # Lout,B,hw,c -> Lout,B,hw,d*cls

        # base + pred
        out_occ = out_occ.view(*out_occ.shape[:-1], self.num_pred_height, self.num_classes).unsqueeze(1) # Lout, inner, bs, h*w, d, num_cls

        return out_occ  
    
    def forward_head_layers(self, next_bev_feats, img_feats=None, img_metas=None):
        """Get freespace estimation from multi-frame BEV feature maps.

        Args:
            next_bev_feats (torch.Tensor): with shape as
                [Lout, inter_num, bs, bev_h * bev_w, dims]    Lout = cur + future_select
        """
        next_bev_preds = []
        crop_gap_boundary_logits = None
        final_level = next_bev_feats.shape[1] - 1
        for lvl in range(next_bev_feats.shape[1]):
            #  ===> Lout, bs, h*w, d, num_frame
            next_bev_pred = self.bev_pred_head[lvl](next_bev_feats[:, lvl]) # C -> d * num_cls
            next_bev_pred = next_bev_pred.view(
                *next_bev_pred.shape[:-1], self.num_pred_height,
                self.num_classes)
            if self.use_gap_residual_refiner and lvl == final_level:
                self._gap_refiner_coarse_logits = next_bev_pred
                next_bev_pred, self._gap_refiner_gate_logits = (
                    self._gap_residual_refine(next_bev_feats[:, lvl],
                                              next_bev_pred, img_feats,
                                              img_metas))
            if self.use_crop_gap_refinement and lvl == final_level:
                next_bev_pred, crop_gap_boundary_logits = (
                    self._crop_gap_refine(next_bev_feats[:, lvl], next_bev_pred))
            if self.use_selective_c2f and lvl == final_level:
                next_bev_pred = self._selective_c2f_refine(
                    next_bev_feats[:, lvl], next_bev_pred)
            next_bev_preds.append(next_bev_pred)
        self._crop_gap_boundary_logits = crop_gap_boundary_logits
        if self.use_dual_hardness_refinement:
            self._dual_hardness_features = next_bev_feats[:, final_level]
        next_bev_preds = torch.stack(next_bev_preds, 1) # Lout, inner, bs, h*w, d, num_cls
        return next_bev_preds
    
    def forward_head(self, next_bev_feats, img_feats=None, img_metas=None):
        self._crop_gap_boundary_logits = None
        self._dual_hardness_features = None
        self._gap_refiner_coarse_logits = None
        self._gap_refiner_gate_logits = None
        if self.soft_weight:
            return self.forward_head_soft(next_bev_feats)   # multi-decoder_layers soft_weight_sum
        else:
            return self.forward_head_layers(
                next_bev_feats, img_feats=img_feats, img_metas=img_metas)

    def _crop_gap_refine(self, features, logits):
        """Gate a voxel-logit residual by crop/free boundary confidence."""
        boundary_logits = self.crop_gap_boundary_head(features)
        residual = self.crop_gap_residual(features).view(
            *features.shape[:-1], self.num_pred_height, self.num_classes)
        probabilities = torch.softmax(logits.float(), dim=-1)
        crop_probability = probabilities[..., self.crop_gap_crop_class]
        free_probability = probabilities[..., 0]
        # Crop/free ambiguity is high only at a putative semantic transition.
        ambiguity = (4.0 * crop_probability * free_probability).to(logits.dtype)
        boundary_gate = torch.sigmoid(boundary_logits).unsqueeze(-1)
        gate = boundary_gate * (0.25 + 0.75 * ambiguity.unsqueeze(-1))
        return logits + gate * residual, boundary_logits

    def _features_to_bev_volume(self, features):
        """Convert ``[T,B,HW,C]`` BEV features into a 3D feature volume."""
        frames, batch, tokens, channels = features.shape
        if tokens != self.bev_h * self.bev_w:
            raise ValueError(
                f'Expected {self.bev_h * self.bev_w} BEV tokens, got {tokens}.')
        feature_map = features.reshape(
            frames * batch, self.bev_h, self.bev_w, channels).permute(
                0, 3, 1, 2).contiguous()
        return self.gap_refiner_bev_proj(feature_map).unsqueeze(2).expand(
            -1, -1, self.num_pred_height, -1, -1)

    def _logits_to_volume(self, logits):
        """Convert public head logits into ``[T*B,C,Z,H,W]``."""
        frames, batch, tokens, depth, classes = logits.shape
        if (tokens, depth, classes) != (
                self.bev_h * self.bev_w, self.num_pred_height,
                self.num_classes):
            raise ValueError('Unexpected occupancy-logit shape for refinement.')
        return logits.permute(0, 1, 4, 3, 2).reshape(
            frames * batch, classes, depth, self.bev_h, self.bev_w)

    def _volume_to_logits(self, volume, frames, batch):
        """Restore ``[T,B,HW,Z,C]`` from ``[T*B,C,Z,H,W]``."""
        return volume.reshape(
            frames, batch, self.num_classes, self.num_pred_height,
            self.bev_h, self.bev_w).permute(0, 1, 4, 5, 3, 2).reshape(
                frames, batch, self.bev_h * self.bev_w,
                self.num_pred_height, self.num_classes)

    @staticmethod
    def _normalized_entropy(probabilities):
        classes = probabilities.shape[1]
        return (-(probabilities * probabilities.clamp_min(1e-6).log()).sum(
            dim=1, keepdim=True) / np.log(float(classes)))

    def _gap_residual_refine(self, features, logits, img_feats=None,
                             img_metas=None):
        """End-to-end BEV/logit conditioned, gated 3D residual refinement."""
        frames, batch = features.shape[:2]
        logit_volume = self._logits_to_volume(logits)
        probabilities = torch.softmax(logit_volume.float(), dim=1)
        crop_probability = probabilities[:, self.crop_gap_crop_class:self.crop_gap_crop_class + 1]
        free_probability = probabilities[:, :1]
        ambiguity = (4.0 * crop_probability * free_probability).to(
            logit_volume.dtype)
        entropy = self._normalized_entropy(probabilities).to(logit_volume.dtype)
        refiner_inputs = [logit_volume, entropy, ambiguity]
        if self.gap_refiner_use_bev_feature:
            refiner_inputs.insert(0, self._features_to_bev_volume(features))
        refine_feature = self.gap_refiner_stem(torch.cat(refiner_inputs, dim=1))
        if self.gap_refiner_use_image_features:
            image_evidence, image_visible = self._gap_image_evidence(
                features, logits, img_feats, img_metas)
            if image_evidence is not None:
                image_feature = self.gap_refiner_image_fuse(torch.cat(
                    (image_evidence, image_visible), dim=1))
                # Current RGB has no counterpart for future predicted frames.
                # The zero-initialized fusion output makes this an identity at
                # initialization, including when temporal prediction is on.
                image_injection = refine_feature.new_zeros(refine_feature.shape)
                image_injection[:batch] = image_feature.to(refine_feature.dtype)
                refine_feature = refine_feature + image_injection
        refine_feature = self.gap_refiner_blocks_module(refine_feature)
        gate_logits = self.gap_refiner_gate(refine_feature)
        residual = self.gap_refiner_delta(refine_feature)
        # Ambiguity prevents a learned gate from needlessly changing confident
        # interiors, while its nonzero floor lets the model correct confidently
        # wrong coarse logits when BEV evidence supports a correction.
        gate = torch.sigmoid(gate_logits) * (0.25 + 0.75 * ambiguity)
        refined = self._volume_to_logits(
            logit_volume + gate * residual, frames, batch)
        gate_logits = gate_logits.reshape(
            frames, batch, 1, self.num_pred_height, self.bev_h,
            self.bev_w).permute(0, 1, 4, 5, 3, 2).squeeze(-1).reshape(
                frames, batch, self.bev_h * self.bev_w,
                self.num_pred_height)
        return refined, gate_logits

    def _gap_image_evidence(self, features, logits, img_feats, img_metas):
        """Return sparse current-image FPN evidence as a dense BEV volume."""
        if img_feats is None or img_metas is None:
            return None, None
        _, batch, tokens, channels = features.shape
        if tokens != self.bev_h * self.bev_w:
            raise ValueError(
                f'GapRef image branch expected {self.bev_h * self.bev_w} BEV '
                f'cells, got {tokens}.')
        active_indices = self._select_gap_image_indices(logits[0])
        sampled, visible = self._sample_gap_image_features(
            img_feats, img_metas, active_indices)
        active_count = active_indices.shape[1]
        refiner_dtype = self.gap_refiner_image_proj.weight.dtype
        image_feature = self.gap_refiner_image_proj(
            sampled.to(refiner_dtype))
        selected_bev = torch.gather(
            features[0], 1,
            active_indices.unsqueeze(-1).expand(-1, -1, channels)).to(
                refiner_dtype)
        view_context = selected_bev[:, None, :, None, :].expand(
            -1, image_feature.shape[1], -1, self.num_pred_height, -1)
        view_scores = self.gap_refiner_image_view_attn(torch.cat(
            (view_context, image_feature), dim=-1)).squeeze(-1)
        view_scores = view_scores.masked_fill(~visible, -1e4)
        view_weight = torch.softmax(view_scores, dim=1) * visible.to(
            view_scores.dtype)
        view_weight = view_weight / view_weight.sum(
            dim=1, keepdim=True).clamp_min(1.0)
        image_feature = (view_weight.unsqueeze(-1) * image_feature).sum(dim=1)
        # Scatter selected [B,K,Z,C] FPN evidence back into [B,C,Z,H,W].
        evidence = image_feature.new_zeros(
            batch, self.gap_refiner_image_channels, self.num_pred_height,
            tokens)
        evidence.scatter_(3, active_indices[:, None, None, :].expand(
            -1, self.gap_refiner_image_channels, self.num_pred_height, -1),
            image_feature.permute(0, 3, 2, 1))
        visibility = image_feature.new_zeros(
            batch, 1, self.num_pred_height, tokens)
        visibility.scatter_(3, active_indices[:, None, None, :].expand(
            -1, 1, self.num_pred_height, -1), visible.any(dim=1).permute(
                0, 2, 1).unsqueeze(1).to(image_feature.dtype))
        return (evidence.reshape(batch, self.gap_refiner_image_channels,
                                 self.num_pred_height, self.bev_h, self.bev_w),
                visibility.reshape(batch, 1, self.num_pred_height,
                                   self.bev_h, self.bev_w))

    def _select_gap_image_indices(self, current_logits):
        """Choose ambiguity cells plus crop interiors with no duplicated cells."""
        batch, tokens = current_logits.shape[:2]
        probabilities = torch.softmax(current_logits.float(), dim=-1)
        crop_probability = probabilities[..., self.crop_gap_crop_class].amax(
            dim=-1)
        free_probability = probabilities[..., 0].amax(dim=-1)
        ambiguity = (4.0 * crop_probability * free_probability)
        active_count = max(1, int(round(
            tokens * self.gap_refiner_image_active_ratio)))
        crop_count = int(round(
            active_count * self.gap_refiner_image_crop_ratio))
        crop_count = min(crop_count, active_count)
        boundary_count = active_count - crop_count
        selected = []
        if boundary_count:
            boundary_indices = torch.topk(
                ambiguity, k=boundary_count, dim=-1, sorted=False).indices
            selected.append(boundary_indices)
        if crop_count:
            crop_scores = crop_probability.clone()
            if boundary_count:
                crop_scores.scatter_(1, boundary_indices, float('-inf'))
            selected.append(torch.topk(
                crop_scores, k=crop_count, dim=-1, sorted=False).indices)
        return torch.cat(selected, dim=1).reshape(batch, active_count)

    def _sample_gap_image_features(self, img_feats, img_metas, active_indices):
        """Project selected voxel centers and preserve all visible camera views."""
        if not isinstance(img_feats, (list, tuple)) or not img_feats:
            raise ValueError('GapRef image branch requires nonempty FPN features.')
        first = img_feats[0]
        if first.dim() != 5:
            raise ValueError('Current FPN features must have shape [B,N,C,H,W].')
        batch, cameras, channels = first.shape[:3]
        if batch != active_indices.shape[0] or len(img_metas) != batch:
            raise ValueError('FPN features, metadata and selected BEV batch differ.')
        if channels != self.embed_dims:
            raise ValueError(
                f'GapRef image branch expects {self.embed_dims}-channel FPN maps, '
                f'got {channels}.')
        if any(len(meta['lidar2img']) != cameras for meta in img_metas):
            raise ValueError('Camera calibration count does not match FPN views.')

        device, dtype = first.device, first.dtype
        active_count = active_indices.shape[1]
        rows = torch.div(active_indices, self.bev_w, rounding_mode='floor')
        cols = active_indices.remainder(self.bev_w)
        x = (self.pc_range[0] + (cols.to(dtype) + 0.5) /
             self.bev_w * (self.pc_range[3] - self.pc_range[0]))
        y = (self.pc_range[1] + (rows.to(dtype) + 0.5) /
             self.bev_h * (self.pc_range[4] - self.pc_range[1]))
        z = torch.linspace(
            self.pc_range[2] + (self.pc_range[5] - self.pc_range[2]) /
            (2 * self.num_pred_height),
            self.pc_range[5] - (self.pc_range[5] - self.pc_range[2]) /
            (2 * self.num_pred_height), self.num_pred_height,
            device=device, dtype=dtype)
        xyz1 = torch.stack((
            x.unsqueeze(-1).expand(-1, -1, self.num_pred_height),
            y.unsqueeze(-1).expand(-1, -1, self.num_pred_height),
            z.view(1, 1, -1).expand(batch, active_count, -1),
            torch.ones(batch, active_count, self.num_pred_height,
                       device=device, dtype=dtype)), dim=-1)
        lidar2img = torch.as_tensor(np.asarray([
            meta['lidar2img'] for meta in img_metas]), device=device,
            dtype=dtype)
        projected = torch.einsum('bnij,bkzj->bnkzi', lidar2img, xyz1)
        depth = projected[..., 2]
        pixels = projected[..., :2] / depth.clamp_min(1e-5).unsqueeze(-1)
        image_size = torch.as_tensor(np.asarray([
            [[shape[1], shape[0]] for shape in meta['img_shape']]
            for meta in img_metas]), device=device, dtype=dtype)
        grid = 2.0 * pixels / (
            image_size[:, :, None, None, :] - 1.0).clamp_min(1.0) - 1.0
        grid = torch.nan_to_num(grid, nan=2.0, posinf=2.0, neginf=-2.0)
        visible = ((depth > 1e-5) & (grid[..., 0] > -1.0) &
                   (grid[..., 0] < 1.0) & (grid[..., 1] > -1.0) &
                   (grid[..., 1] < 1.0))

        sampled_levels = []
        for feature in img_feats[:self.gap_refiner_image_levels]:
            if feature.shape[:3] != (batch, cameras, channels):
                raise ValueError('All FPN levels must share [B,N,C] dimensions.')
            flattened_feature = feature.reshape(
                batch * cameras, channels, feature.shape[-2], feature.shape[-1])
            flattened_grid = grid.reshape(
                batch * cameras, active_count * self.num_pred_height, 1, 2)
            sampled = F.grid_sample(
                flattened_feature, flattened_grid.to(feature.dtype),
                mode='bilinear', padding_mode='zeros', align_corners=True)
            sampled = sampled.reshape(
                batch, cameras, channels, active_count,
                self.num_pred_height).permute(0, 1, 3, 4, 2)
            sampled_levels.append(sampled)
        return torch.stack(sampled_levels, dim=0).mean(dim=0), visible

    def _selective_c2f_refine(self, features, logits):
        """Decode 2x2 child subqueries for only uncertain crop/free cells.

        The public occupancy output remains at the configured BEV resolution.
        Child predictions are averaged into a residual for their parent cell,
        so this stays compatible with the existing 100x100 FarmSim labels and
        evaluator while giving difficult cells a higher-capacity decoder.
        """
        frames, batch, tokens, channels = features.shape
        if tokens != self.bev_h * self.bev_w:
            raise ValueError(
                f'Selective C2F expected {self.bev_h * self.bev_w} BEV cells, '
                f'got {tokens}.')
        probabilities = torch.softmax(logits.float(), dim=-1)
        crop_probability = probabilities[..., self.crop_gap_crop_class]
        free_probability = probabilities[..., 0]
        difficulty = (4.0 * crop_probability * free_probability).amax(dim=-1)
        active_count = max(1, int(round(tokens * self.c2f_active_ratio)))
        active_indices = torch.topk(
            difficulty, k=active_count, dim=-1, sorted=False).indices
        selected = torch.gather(
            features, 2,
            active_indices.unsqueeze(-1).expand(-1, -1, -1, channels))
        child_features = selected.unsqueeze(-2) + self.c2f_subquery_offsets.view(
            1, 1, 1, 4, channels)
        child_residuals = self.c2f_subquery_decoder(child_features).view(
            frames, batch, active_count, 4, self.num_pred_height,
            self.num_classes)
        parent_residuals = child_residuals.mean(dim=3)
        residual = logits.new_zeros(logits.shape)
        residual.scatter_(
            2, active_indices.unsqueeze(-1).unsqueeze(-1).expand(
                -1, -1, -1, self.num_pred_height, self.num_classes),
            parent_residuals)
        return logits + residual

    @staticmethod
    def _shift_voxel_mask(mask, dh, dw, dz):
        """Read a zero-padded 3D neighbour from a ``[B,H,W,Z]`` mask."""
        height, width, depth = mask.shape[-3:]
        padded = F.pad(mask.unsqueeze(1).float(), (1, 1, 1, 1, 1, 1))
        return padded[:, 0, 1 + dh:1 + dh + height,
                      1 + dw:1 + dw + width,
                      1 + dz:1 + dz + depth].bool()

    @staticmethod
    def _shift_voxel_values(values, dh, dw, dz):
        """Read a zero-padded 3D neighbour from ``[B,H,W,Z]`` values."""
        height, width, depth = values.shape[-3:]
        padded = F.pad(values.unsqueeze(1), (1, 1, 1, 1, 1, 1))
        return padded[:, 0, 1 + dh:1 + dh + height,
                      1 + dw:1 + dw + width,
                      1 + dz:1 + dz + depth]

    def _crop_free_boundary_mask(self, target_voxels):
        """Derive a crop/free transition mask directly from semantic GT."""
        known = target_voxels != 255
        crop = (target_voxels == self.crop_gap_crop_class) & known
        free = (target_voxels == 0) & known
        boundary = torch.zeros_like(known)
        for dh, dw, dz in ((1, 0, 0), (-1, 0, 0), (0, 1, 0),
                           (0, -1, 0), (0, 0, 1), (0, 0, -1)):
            neighbour_known = self._shift_voxel_mask(known, dh, dw, dz)
            neighbour_crop = self._shift_voxel_mask(crop, dh, dw, dz)
            neighbour_free = self._shift_voxel_mask(free, dh, dw, dz)
            boundary |= neighbour_known & (
                (crop & neighbour_free) | (free & neighbour_crop))
        return boundary

    def _near_crop_free_weights(self, target_voxels):
        """Return distance-decayed weights for free voxels close to crop."""
        known = target_voxels != 255
        crop = ((target_voxels == self.crop_gap_crop_class) & known)
        free = ((target_voxels == 0) & known)
        reached = crop.unsqueeze(1).float()
        distance = target_voxels.new_full(
            target_voxels.shape, self.crop_gap_radius + 1,
            dtype=torch.float32)
        for step in range(self.crop_gap_radius + 1):
            if step:
                reached = F.max_pool3d(reached, kernel_size=3, stride=1,
                                       padding=1)
            newly_reached = reached[:, 0].bool() & (distance > self.crop_gap_radius)
            distance = torch.where(
                newly_reached, distance.new_full(distance.shape, float(step)),
                distance)
        near_crop_free = free & (distance <= self.crop_gap_radius)
        weights = 1.0 + self.crop_gap_alpha * torch.exp(
            -distance / self.crop_gap_sigma)
        return near_crop_free, weights

    def _near_free_crop_weights(self, target_voxels):
        """Return distance-decayed weights for crop voxels close to free.

        This is the counterpart of ``_near_crop_free_weights``.  Training the
        refiner with both masks prevents a free-gap objective from lowering
        its loss by eroding small crop regions around plant boundaries.
        """
        known = target_voxels != 255
        crop = ((target_voxels == self.crop_gap_crop_class) & known)
        free = ((target_voxels == 0) & known)
        reached = free.unsqueeze(1).float()
        distance = target_voxels.new_full(
            target_voxels.shape, self.crop_gap_radius + 1,
            dtype=torch.float32)
        for step in range(self.crop_gap_radius + 1):
            if step:
                reached = F.max_pool3d(reached, kernel_size=3, stride=1,
                                       padding=1)
            newly_reached = reached[:, 0].bool() & (
                distance > self.crop_gap_radius)
            distance = torch.where(
                newly_reached, distance.new_full(distance.shape, float(step)),
                distance)
        near_free_crop = crop & (distance <= self.crop_gap_radius)
        weights = 1.0 + self.crop_gap_alpha * torch.exp(
            -distance / self.crop_gap_sigma)
        return near_free_crop, weights

    def _local_semantic_anisotropy(self, target_voxels):
        """Count valid six-neighbour semantic changes for every voxel."""
        known = target_voxels != 255
        anisotropy = target_voxels.new_zeros(target_voxels.shape,
                                              dtype=torch.float32)
        for dh, dw, dz in ((1, 0, 0), (-1, 0, 0), (0, 1, 0),
                           (0, -1, 0), (0, 0, 1), (0, 0, -1)):
            neighbour_known = self._shift_voxel_mask(known, dh, dw, dz)
            neighbour = self._shift_voxel_values(target_voxels, dh, dw, dz)
            anisotropy += (known & neighbour_known &
                            (target_voxels != neighbour)).to(anisotropy.dtype)
        return anisotropy

    @torch.no_grad()
    def _update_dual_hardness_teacher(self):
        """Advance the ADHR EMA teacher once per training loss computation."""
        if not self.training:
            return
        decay = self.dual_hardness_ema_decay
        for teacher, student in zip(self.dual_hardness_teacher_z_embed.parameters(),
                                    self.dual_hardness_z_embed.parameters()):
            teacher.mul_(decay).add_(student, alpha=1.0 - decay)
        for teacher, student in zip(self.dual_hardness_teacher.parameters(),
                                    self.dual_hardness_refiner.parameters()):
            teacher.mul_(decay).add_(student, alpha=1.0 - decay)

    def _dual_hardness_indices(self, logits, target_voxels):
        """Select uncertain voxels while reserving a crop/free-gap quota."""
        batch, _, height, width, depth = logits.shape
        total_voxels = height * width * depth
        selected_count = max(1, int(round(
            total_voxels * self.dual_hardness_active_ratio)))
        probabilities = torch.softmax(logits.float(), dim=1)
        top_two = probabilities.topk(k=2, dim=1).values
        global_hardness = 1.0 / (top_two[:, 0] - top_two[:, 1]).clamp_min(1e-3)
        known = target_voxels != 255
        crop_gap = (self._crop_free_boundary_mask(target_voxels) |
                    self._near_crop_free_weights(target_voxels)[0])
        flat_hardness = global_hardness.reshape(batch, -1)
        flat_known = known.reshape(batch, -1)
        flat_gap = crop_gap.reshape(batch, -1)
        selected_indices, selected_valid = [], []
        desired_gap = int(round(selected_count * self.dual_hardness_gap_ratio))
        for batch_index in range(batch):
            valid_indices = torch.nonzero(flat_known[batch_index],
                                          as_tuple=False).flatten()
            if not len(valid_indices):
                selected_indices.append(torch.zeros(
                    selected_count, dtype=torch.long, device=logits.device))
                selected_valid.append(torch.zeros(
                    selected_count, dtype=torch.bool, device=logits.device))
                continue
            gap_indices = torch.nonzero(
                flat_gap[batch_index] & flat_known[batch_index],
                as_tuple=False).flatten()
            gap_count = min(desired_gap, len(gap_indices), selected_count)
            if gap_count:
                gap_scores = flat_hardness[batch_index, gap_indices]
                chosen_gap = gap_indices[torch.topk(
                    gap_scores, k=gap_count, sorted=False).indices]
            else:
                chosen_gap = valid_indices.new_empty((0,))
            available = flat_known[batch_index].clone()
            available[chosen_gap] = False
            remaining_indices = torch.nonzero(available, as_tuple=False).flatten()
            remaining_count = min(selected_count - gap_count,
                                  len(remaining_indices))
            if remaining_count:
                remaining_scores = flat_hardness[batch_index, remaining_indices]
                chosen_remaining = remaining_indices[torch.topk(
                    remaining_scores, k=remaining_count, sorted=False).indices]
            else:
                chosen_remaining = valid_indices.new_empty((0,))
            chosen = torch.cat((chosen_gap, chosen_remaining), dim=0)
            is_valid = torch.ones(len(chosen), dtype=torch.bool,
                                  device=logits.device)
            if len(chosen) < selected_count:
                pad_count = selected_count - len(chosen)
                chosen = torch.cat((chosen, valid_indices[:1].expand(pad_count)))
                is_valid = torch.cat((is_valid, torch.zeros(
                    pad_count, dtype=torch.bool, device=logits.device)))
            selected_indices.append(chosen)
            selected_valid.append(is_valid)
        return torch.stack(selected_indices), torch.stack(selected_valid), crop_gap

    def loss_dual_hardness_refinement(self, output_voxels, target_voxels):
        """ADHR: supervised hard-voxel refinement with an EMA teacher."""
        if (not self.use_dual_hardness_refinement or
                self._dual_hardness_features is None):
            return {}
        logits = output_voxels[-1]
        target_voxels = target_voxels.long()
        if logits.shape[-3:] != target_voxels.shape[-3:]:
            return {'loss_adhr_refine': logits.sum() * 0,
                    'loss_adhr_distill': logits.sum() * 0}
        features = self._dual_hardness_features.reshape(
            -1, self.bev_h * self.bev_w, self.embed_dims)
        if features.shape[0] != target_voxels.shape[0]:
            return {'loss_adhr_refine': logits.sum() * 0,
                    'loss_adhr_distill': logits.sum() * 0}
        indices, selected_valid, crop_gap = self._dual_hardness_indices(
            logits, target_voxels)
        batch, _, _, width, depth = logits.shape
        selected_labels = torch.gather(target_voxels.reshape(batch, -1), 1,
                                       indices)
        cell_indices = torch.div(indices, depth, rounding_mode='floor')
        height_indices = torch.div(cell_indices, width, rounding_mode='floor')
        width_indices = cell_indices.remainder(width)
        z_indices = indices.remainder(depth)
        bev_indices = height_indices * width + width_indices
        selected_features = torch.gather(
            features, 1,
            bev_indices.unsqueeze(-1).expand(-1, -1, self.embed_dims))
        student_features = selected_features + self.dual_hardness_z_embed(z_indices)
        semantic_logits = logits.permute(0, 2, 3, 4, 1).reshape(
            batch, -1, self.num_classes)
        coarse_selected = torch.gather(
            semantic_logits, 1,
            indices.unsqueeze(-1).expand(-1, -1, self.num_classes))
        refined_logits = coarse_selected + self.dual_hardness_refiner(
            student_features)
        anisotropy = self._local_semantic_anisotropy(target_voxels).reshape(batch, -1)
        selected_anisotropy = torch.gather(anisotropy, 1, indices)
        selected_gap = torch.gather(crop_gap.reshape(batch, -1), 1, indices)
        weights = (1.0 + self.dual_hardness_local_scale * selected_anisotropy +
                   self.dual_hardness_gap_boost * selected_gap.to(
                       selected_anisotropy.dtype))
        valid = selected_valid & (selected_labels != 255)
        if valid.any():
            refine_loss = F.cross_entropy(
                refined_logits.transpose(1, 2), selected_labels,
                ignore_index=255, reduction='none')
            refine_loss = (refine_loss[valid] * weights[valid]).sum() / (
                weights[valid].sum().clamp_min(1.0))
        else:
            refine_loss = logits.sum() * 0

        self._update_dual_hardness_teacher()
        with torch.no_grad():
            teacher_features = selected_features.detach() + \
                self.dual_hardness_teacher_z_embed(z_indices)
            teacher_logits = coarse_selected.detach() + self.dual_hardness_teacher(
                teacher_features)
        if valid.any():
            temperature = 2.0
            distill = F.kl_div(
                F.log_softmax(refined_logits / temperature, dim=-1),
                F.softmax(teacher_logits / temperature, dim=-1),
                reduction='none').sum(dim=-1) * (temperature ** 2)
            distill = (distill[valid] * weights[valid]).sum() / (
                weights[valid].sum().clamp_min(1.0))
        else:
            distill = logits.sum() * 0
        return {
            'loss_adhr_refine': self.dual_hardness_loss_weight * refine_loss,
            'loss_adhr_distill': self.dual_hardness_distill_weight * distill,
        }

    def _raw_logits_to_loss_volume(self, logits):
        """Match Drive_OccWorld_V2's public-logit to loss-volume reshape."""
        frames, batch, tokens, depth, classes = logits.shape
        if tokens != self.bev_h * self.bev_w:
            raise ValueError('Unexpected BEV token count in cached coarse logits.')
        return logits.permute(0, 1, 4, 2, 3).reshape(
            frames * batch, classes, self.bev_w, self.bev_h, depth).transpose(
                2, 3).contiguous()

    @staticmethod
    def _scale_loss_dict(loss_dict, scale):
        return {name: value * scale for name, value in loss_dict.items()}

    def _gap_free_loss(self, logits, target_voxels, weight):
        """Penalize filling valid free voxels that lie close to crop GT."""
        near_crop_free, gap_weights = self._near_crop_free_weights(
            target_voxels)
        if near_crop_free.any():
            free_negative_log_likelihood = -F.log_softmax(logits, dim=1)[:, 0]
            loss = (free_negative_log_likelihood[near_crop_free] *
                    gap_weights.to(logits.dtype)[near_crop_free]).sum()
            loss = loss / gap_weights[near_crop_free].sum().clamp_min(1.0)
        else:
            loss = logits.sum() * 0
        return weight * loss

    def _gap_crop_loss(self, logits, target_voxels, weight):
        """Protect small crop voxels adjacent to free inter-plant gaps."""
        near_free_crop, crop_weights = self._near_free_crop_weights(
            target_voxels)
        if near_free_crop.any():
            crop_negative_log_likelihood = -F.log_softmax(
                logits, dim=1)[:, self.crop_gap_crop_class]
            loss = (crop_negative_log_likelihood[near_free_crop] *
                    crop_weights.to(logits.dtype)[near_free_crop]).sum()
            loss = loss / crop_weights[near_free_crop].sum().clamp_min(1.0)
        else:
            loss = logits.sum() * 0
        return weight * loss

    def loss_gap_residual_refiner(self, output_voxels, target_voxels):
        """Deep coarse supervision plus crop-gap objectives for proposal one."""
        if not self.use_gap_residual_refiner:
            return {}
        coarse_logits = self._gap_refiner_coarse_logits
        if coarse_logits is None:
            return {'loss_gap_refiner_coarse': output_voxels.sum() * 0}
        coarse_volume = self._raw_logits_to_loss_volume(coarse_logits)
        loss_dict = self._scale_loss_dict(
            self.loss_voxel(coarse_volume, target_voxels,
                            tag='gap_refiner_coarse'),
            self.gap_refiner_coarse_loss_weight)
        refined_logits = output_voxels[-1]
        if refined_logits.shape[-3:] != target_voxels.shape[-3:]:
            refined_logits = F.interpolate(
                refined_logits, size=target_voxels.shape[-3:],
                mode='trilinear', align_corners=False)
        target_voxels = target_voxels.long()
        known = target_voxels != 255
        boundary_target = self._crop_free_boundary_mask(target_voxels)
        gate_logits = self._gap_refiner_gate_logits
        if gate_logits is None:
            boundary_loss = refined_logits.sum() * 0
        else:
            gate_logits = gate_logits.reshape(
                -1, self.bev_w, self.bev_h, self.num_pred_height).transpose(
                    1, 2).contiguous()
            if gate_logits.shape[-3:] != target_voxels.shape[-3:]:
                gate_logits = F.interpolate(
                    gate_logits.unsqueeze(1), size=target_voxels.shape[-3:],
                    mode='trilinear', align_corners=False).squeeze(1)
            if known.any():
                gate_weight = 1.0 + 3.0 * boundary_target.to(
                    refined_logits.dtype)
                boundary_loss = F.binary_cross_entropy_with_logits(
                    gate_logits[known], boundary_target[known].to(
                        refined_logits.dtype), weight=gate_weight[known])
            else:
                boundary_loss = refined_logits.sum() * 0
        loss_dict['loss_gap_refiner_boundary'] = (
            self.gap_refiner_boundary_loss_weight * boundary_loss)
        loss_dict['loss_gap_refiner_free'] = self._gap_free_loss(
            refined_logits, target_voxels, self.gap_refiner_gap_loss_weight)
        loss_dict['loss_gap_refiner_crop'] = self._gap_crop_loss(
            refined_logits, target_voxels, self.gap_refiner_crop_loss_weight)
        return loss_dict

    def loss_crop_gap_refinement(self, output_voxels, target_voxels):
        """Supervise crop/free boundaries and preserve close free gaps."""
        if not self.use_crop_gap_refinement:
            return {}
        logits = output_voxels[-1]
        target_voxels = target_voxels.long()
        if logits.shape[-3:] != target_voxels.shape[-3:]:
            logits = F.interpolate(logits, size=target_voxels.shape[-3:],
                                   mode='trilinear', align_corners=False)
        known = target_voxels != 255
        boundary_target = self._crop_free_boundary_mask(target_voxels)
        loss_dict = {}

        boundary_logits = self._crop_gap_boundary_logits
        expected_batch = target_voxels.shape[0]
        if (boundary_logits is None or
                boundary_logits.shape[0] * boundary_logits.shape[1] != expected_batch):
            loss_dict['loss_crop_gap_boundary'] = logits.sum() * 0
        else:
            boundary_logits = boundary_logits.reshape(
                expected_batch, self.bev_h, self.bev_w,
                self.num_pred_height).unsqueeze(1)
            if boundary_logits.shape[-3:] != target_voxels.shape[-3:]:
                boundary_logits = F.interpolate(
                    boundary_logits, size=target_voxels.shape[-3:],
                    mode='trilinear', align_corners=False)
            boundary_logits = boundary_logits[:, 0]
            if known.any():
                # Balance the sparse positive boundary target without adding
                # another dataset annotation or a global hard-mining pass.
                sample_weight = 1.0 + 3.0 * boundary_target.to(logits.dtype)
                boundary_loss = F.binary_cross_entropy_with_logits(
                    boundary_logits[known], boundary_target[known].to(logits.dtype),
                    weight=sample_weight[known])
            else:
                boundary_loss = logits.sum() * 0
            loss_dict['loss_crop_gap_boundary'] = (
                self.crop_gap_boundary_loss_weight * boundary_loss)

        near_crop_free, gap_weights = self._near_crop_free_weights(target_voxels)
        if near_crop_free.any():
            free_negative_log_likelihood = -F.log_softmax(logits, dim=1)[:, 0]
            gap_loss = (free_negative_log_likelihood[near_crop_free] *
                        gap_weights.to(logits.dtype)[near_crop_free]).sum()
            gap_loss = gap_loss / gap_weights[near_crop_free].sum().clamp_min(1.0)
        else:
            gap_loss = logits.sum() * 0
        loss_dict['loss_crop_gap_free'] = self.crop_gap_free_loss_weight * gap_loss
        return loss_dict

    def loss_voxel(self, output_voxels, target_voxels, tag):
        B, C, pH, pW, pD = output_voxels.shape
        tB, tH, tW, tD = target_voxels.shape
        # FarmSim labels use class 0 for free space.  This must be available
        # even when the prediction grid already matches the target grid.
        empty_idx = 0

        # Targets are not necessarily the fixed nuScenes 256x256x20 grid.
        H, W, D = target_voxels.shape[-3:]

        # output_voxel align to H,W,D
        if pH != H:
            output_voxels = F.interpolate(output_voxels, size=(H, W, D), mode='trilinear', align_corners=False)

        # target_voxel align to H,W,D
        ratio = tH // H
        if ratio != 1:
            target_voxels = target_voxels.reshape(B, H, ratio, W, ratio, D, ratio).permute(0,1,3,5,2,4,6).reshape(B, H, W, D, ratio**3)
            empty_mask = target_voxels.sum(-1) == empty_idx    # B,H,W,D
            target_voxels = target_voxels.to(torch.int64)
            occ_space = target_voxels[~empty_mask]
            occ_space[occ_space==0] = -torch.arange(len(occ_space[occ_space==0])).to(occ_space.device) - 1
            target_voxels[~empty_mask] = occ_space
            target_voxels = torch.mode(target_voxels, dim=-1)[0]
            target_voxels[target_voxels<0] = 255
            target_voxels = target_voxels.long()

        assert torch.isnan(output_voxels).sum().item() == 0
        assert torch.isnan(target_voxels).sum().item() == 0

        loss_dict = {}

        loss_dict['loss_voxel_ce_{}'.format(tag)] = self.loss_voxel_ce_weight * CE_ssc_loss(output_voxels, target_voxels, self.class_weights.type_as(output_voxels), ignore_index=255)

        if self.multi_loss:
            loss_dict['loss_voxel_sem_scal_{}'.format(tag)] = self.loss_voxel_sem_scal_weight * sem_scal_loss(output_voxels, target_voxels, ignore_index=255)
            loss_dict['loss_voxel_lovasz_{}'.format(tag)] = self.loss_voxel_lovasz_weight * lovasz_softmax(torch.softmax(output_voxels, dim=1), target_voxels, ignore=255)
            if self.loss_voxel_geo_scal_weight is not None:
                loss_dict['loss_voxel_geo_scal_{}'.format(tag)] = self.loss_voxel_geo_scal_weight * geo_scal_loss(output_voxels, target_voxels, ignore_index=255, non_empty_idx=empty_idx)

        return loss_dict
    
    def loss_occ(self, output_voxels=None, target_voxels=None, **kwargs):
        """
            output_voxels = inter_num, select_frame*bs, cls, h,w,d
            target_voxels =            select_frame*bs,      H,W,D
        """
        loss_dict = {}
        for index, output_voxel in enumerate(output_voxels):
            loss_dict.update(self.loss_voxel(output_voxel, target_voxels,  tag='inter_{}'.format(index)))
        loss_dict.update(self.loss_crop_gap_refinement(output_voxels, target_voxels))
        loss_dict.update(self.loss_dual_hardness_refinement(
            output_voxels, target_voxels))
        loss_dict.update(self.loss_gap_residual_refiner(
            output_voxels, target_voxels))

        return loss_dict
    
    def loss_sem_norm(self, output_voxels=None, target_voxels=None, **kwargs):
        """
            output_voxels = inter_num, select_frame*bs, cls, h,w,d
            target_voxels =            select_frame*bs,      H,W,D
        """
        inter, B, C, pH, pW, pD = output_voxels.shape
        tB, tH, tW, tD = target_voxels.shape

        H, W, D = target_voxels.shape[-3:]
        # output_voxel align to H,W,D
        if pH != H:
            output_voxels = F.interpolate(output_voxels.flatten(0,1), size=(H, W, D), mode='trilinear', align_corners=False)
            output_voxels = output_voxels.view(inter, B,C,H,W,D)
        
        # target_voxel align to H,W,D
        ratio = tH // H
        if ratio != 1:
            target_voxels = target_voxels.reshape(B, H, ratio, W, ratio, D, ratio).permute(0,1,3,5,2,4,6).reshape(B, H, W, D, ratio**3)
            empty_idx = 0
            empty_mask = target_voxels.sum(-1) == empty_idx
            target_voxels = target_voxels.to(torch.int64)
            occ_space = target_voxels[~empty_mask]
            occ_space[occ_space==0] = -torch.arange(len(occ_space[occ_space==0])).to(occ_space.device) - 1
            target_voxels[~empty_mask] = occ_space
            target_voxels = torch.mode(target_voxels, dim=-1)[0]
            target_voxels[target_voxels<0] = 255
            target_voxels = target_voxels.long()
        
        assert torch.isnan(output_voxels).sum().item() == 0
        assert torch.isnan(target_voxels).sum().item() == 0

        
        loss_dict = {}
        for index, output_voxel in enumerate(output_voxels):
            inter_loss = CE_ssc_loss(output_voxel, target_voxels, self.class_weights.type_as(output_voxels), ignore_index=255)
            loss_dict['loss_sem_norm_{}'.format(index)] = inter_loss

        return loss_dict
