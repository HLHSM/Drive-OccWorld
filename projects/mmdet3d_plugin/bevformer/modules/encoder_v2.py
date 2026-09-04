import torch
import copy
import warnings
from mmcv.cnn.bricks.registry import (ATTENTION,
                                      TRANSFORMER_LAYER,
                                      TRANSFORMER_LAYER_SEQUENCE)
from mmcv.cnn.bricks.transformer import build_attention
from mmdet.models.utils.builder import TRANSFORMER

from .custom_base_transformer_layer import MyCustomBaseTransformerLayer
from .encoder import BEVFormerEncoder
from .transformer import PerceptionTransformer
from .conditionalnorm import ConditionalNorm


@TRANSFORMER.register_module()
class CustomPerceptionTransformer(PerceptionTransformer):

    def init_weights(self):
        """Initialize the transformer weights."""
        for m in self.modules():
            if isinstance(m, ConditionalNorm):
                m.init_weights()


@TRANSFORMER_LAYER_SEQUENCE.register_module()
class CustomBEVFormerEncoder(BEVFormerEncoder):
    def __init__(self,
                 keep_idx=(2,),
                 use_nearfar_bev=False,
                 nearfar_near_ratio=0.6,
                 nearfar_far_stride=2,
                 *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.use_nearfar_bev = bool(use_nearfar_bev)
        self.nearfar_near_ratio = float(nearfar_near_ratio)
        self.nearfar_far_stride = int(nearfar_far_stride)
        if not 0.0 < self.nearfar_near_ratio <= 1.0:
            raise ValueError('nearfar_near_ratio must be in (0, 1].')
        if self.nearfar_far_stride < 2:
            raise ValueError('nearfar_far_stride must be at least 2.')
        if self.use_nearfar_bev:
            if len(self.layers) < 2:
                raise ValueError('Near-far BEV requires at least two encoder layers.')

        self.keep_idx = keep_idx
        # remove latent rendering in previous layers.
        for lid, layer in enumerate(self.layers):
            if lid not in self.keep_idx:
                # if this is not the last layer, and remove operations in previous layers.
                if getattr(layer, 'latent_render', None):
                    del layer.latent_render
                    layer.operation_order = ('self_attn', 'norm', 'cross_attn', 'norm', 'ffn', 'norm')

    def forward(self, *args, **kwargs):
        # Do not decorate this ``*args/**kwargs`` wrapper with ``auto_fp16``.
        # MMCV 1.4 derives the positional argument names from the wrapper
        # signature and consequently drops all positional tensors before the
        # parent call.  BEVFormerEncoder.forward has an explicit signature and
        # is already decorated, so it performs the FP16 cast/autocast safely.
        default_return_intermediate = self.return_intermediate
        self.return_intermediate = kwargs.get('return_intermediate', self.return_intermediate)
        ret = super().forward(*args, **kwargs)
        self.return_intermediate = default_return_intermediate
        return ret

    def _nearfar_layout(self, bev_h, bev_w, device):
        """Return forward-x sparse queries and bilinear restore metadata.

        BEVFormer stores y along H and x along W.  FarmSim's x axis is
        forward, so the near/far boundary must split columns, not rows.  The
        near columns retain all queries; far columns retain a regular 2D grid.
        Missing far queries are bilinearly restored from their four adjacent
        active tokens before an image-conditioned dense refinement layer.
        """
        near_cols = min(bev_w, max(1, int(round(
            bev_w * self.nearfar_near_ratio))))
        ids = torch.arange(bev_h * bev_w, device=device).reshape(bev_h, bev_w)
        active_mask = torch.zeros((bev_h, bev_w), dtype=torch.bool,
                                  device=device)
        active_mask[:, :near_cols] = True
        if near_cols < bev_w:
            active_mask[::self.nearfar_far_stride,
                        near_cols::self.nearfar_far_stride] = True
        active_indices = ids[active_mask]
        active_positions = torch.full((bev_h * bev_w,), -1, dtype=torch.long,
                                      device=device)
        active_positions[active_indices] = torch.arange(
            active_indices.numel(), device=device)
        if near_cols == bev_w:
            restore_positions = active_positions[ids].reshape(-1, 1).expand(
                -1, 4)
            restore_weights = torch.zeros((bev_h * bev_w, 4), device=device)
            restore_weights[:, 0] = 1
            return active_indices, restore_positions, restore_weights

        row_ids = torch.arange(bev_h, device=device).view(-1, 1).expand(
            bev_h, bev_w)
        col_ids = torch.arange(bev_w, device=device).view(1, -1).expand(
            bev_h, bev_w)
        stride = self.nearfar_far_stride
        max_active_row = ((bev_h - 1) // stride) * stride
        max_active_col = near_cols + ((bev_w - 1 - near_cols) // stride) * stride
        row_floor = (row_ids // stride) * stride
        row_ceil = (row_floor + stride).clamp(max=max_active_row)
        col_floor = near_cols + ((col_ids - near_cols).clamp_min(0) // stride) * stride
        col_ceil = (col_floor + stride).clamp(max=max_active_col)
        corner_ids = torch.stack((
            row_floor * bev_w + col_floor,
            row_floor * bev_w + col_ceil,
            row_ceil * bev_w + col_floor,
            row_ceil * bev_w + col_ceil,
        ), dim=-1)
        restore_positions = active_positions[corner_ids.reshape(-1)].reshape(
            bev_h, bev_w, 4)
        row_fraction = ((row_ids - row_floor).to(torch.float32) / stride)
        col_fraction = ((col_ids - col_floor).to(torch.float32) / stride)
        restore_weights = torch.stack((
            (1 - row_fraction) * (1 - col_fraction),
            (1 - row_fraction) * col_fraction,
            row_fraction * (1 - col_fraction),
            row_fraction * col_fraction,
        ), dim=-1)
        near_mask = col_ids < near_cols
        identity_positions = active_positions[ids]
        restore_positions = torch.where(
            near_mask.unsqueeze(-1),
            identity_positions.unsqueeze(-1).expand_as(restore_positions),
            restore_positions)
        restore_weights = torch.where(
            near_mask.unsqueeze(-1),
            torch.tensor((1., 0., 0., 0.), device=device).view(1, 1, 4),
            restore_weights)
        restore_positions = restore_positions.reshape(-1, 4)
        restore_weights = restore_weights.reshape(-1, 4)
        if (restore_positions < 0).any():
            raise RuntimeError('Near-far BEV layout has an unassigned restore cell.')
        return active_indices, restore_positions, restore_weights

    def _forward_nearfar(self, bev_query, key, value, bev_h, bev_w, bev_pos,
                         spatial_shapes, level_start_index, prev_bev, shift,
                         img_metas, *args, **kwargs):
        """Apply full image attention near-field and on a far-field grid."""
        dense_query = bev_query.permute(1, 0, 2)
        dense_pos = bev_pos.permute(1, 0, 2)
        dense_prev_bev = prev_bev
        active_indices, restore_positions, restore_weights = self._nearfar_layout(
            bev_h, bev_w, bev_query.device)
        active_query = dense_query.index_select(1, active_indices)
        active_pos = dense_pos.index_select(1, active_indices)
        batch_size, active_count, _ = active_query.shape
        ref_3d = self.get_reference_points(
            bev_h, bev_w, self.pc_range[5] - self.pc_range[2],
            self.num_points_in_pillar, dim='3d', bs=batch_size,
            device=bev_query.device, dtype=bev_query.dtype).index_select(
                2, active_indices)
        ref_2d = self.get_reference_points(
            bev_h, bev_w, dim='2d', bs=batch_size, device=bev_query.device,
            dtype=bev_query.dtype).index_select(1, active_indices)
        reference_points_cam, bev_mask = self.point_sampling(
            ref_3d, self.pc_range, img_metas)
        shifted_ref_2d = ref_2d.clone()
        shifted_ref_2d += shift[:, None, None, :]
        if dense_prev_bev is not None:
            sparse_prev_bev = dense_prev_bev.index_select(
                0, active_indices).permute(1, 0, 2)
            sparse_prev_bev = torch.stack([sparse_prev_bev, active_query], 1).reshape(
                batch_size * 2, active_count, -1)
            hybrid_ref_2d = torch.stack([shifted_ref_2d, ref_2d], 1).reshape(
                batch_size * 2, active_count, 1, 2)
        else:
            sparse_prev_bev = None
            hybrid_ref_2d = torch.stack([ref_2d, ref_2d], 1).reshape(
                batch_size * 2, active_count, 1, 2)

        output = active_query
        # Most layers operate on the geometry-selected tokens.  The final
        # dense layer restores direct image cross-attention to every BEV cell,
        # avoiding the old query-only completion of far-field semantics.
        for layer in self.layers[:-1]:
            output = layer(
                output, key, value, *args, bev_pos=active_pos,
                ref_2d=hybrid_ref_2d, ref_3d=ref_3d,
                bev_h=active_count, bev_w=1, spatial_shapes=spatial_shapes,
                level_start_index=level_start_index,
                reference_points_cam=reference_points_cam, bev_mask=bev_mask,
                prev_bev=sparse_prev_bev, gvad_sparse_layout=True,
                gvad_active_indices=active_indices, gvad_dense_bev_h=bev_h,
                gvad_dense_bev_w=bev_w, **kwargs)
        restored = output[:, restore_positions]
        restored = (restored * restore_weights.to(output.dtype).view(
            1, -1, 4, 1)).sum(dim=2)

        dense_ref_3d = self.get_reference_points(
            bev_h, bev_w, self.pc_range[5] - self.pc_range[2],
            self.num_points_in_pillar, dim='3d', bs=batch_size,
            device=bev_query.device, dtype=bev_query.dtype)
        dense_ref_2d = self.get_reference_points(
            bev_h, bev_w, dim='2d', bs=batch_size, device=bev_query.device,
            dtype=bev_query.dtype)
        dense_reference_points_cam, dense_bev_mask = self.point_sampling(
            dense_ref_3d, self.pc_range, img_metas)
        dense_shifted_ref_2d = dense_ref_2d.clone()
        dense_shifted_ref_2d += shift[:, None, None, :]
        if dense_prev_bev is not None:
            dense_prev = dense_prev_bev.permute(1, 0, 2)
            dense_prev = torch.stack([dense_prev, restored], 1).reshape(
                batch_size * 2, bev_h * bev_w, -1)
            dense_hybrid_ref_2d = torch.stack(
                [dense_shifted_ref_2d, dense_ref_2d], 1).reshape(
                    batch_size * 2, bev_h * bev_w, 1, 2)
        else:
            dense_prev = None
            dense_hybrid_ref_2d = torch.stack(
                [dense_ref_2d, dense_ref_2d], 1).reshape(
                    batch_size * 2, bev_h * bev_w, 1, 2)
        return self.layers[-1](
            restored, key, value, *args, bev_pos=dense_pos,
            ref_2d=dense_hybrid_ref_2d, ref_3d=dense_ref_3d,
            bev_h=bev_h, bev_w=bev_w, spatial_shapes=spatial_shapes,
            level_start_index=level_start_index,
            reference_points_cam=dense_reference_points_cam,
            bev_mask=dense_bev_mask, prev_bev=dense_prev,
            gvad_sparse_layout=False, **kwargs)


@TRANSFORMER_LAYER.register_module()
class BEVFormerLayerV2(MyCustomBaseTransformerLayer):
    """BEVFormerLayerV2, enhanced with ray-aware deformable attention.

    The corresponding ray-aware attention layer is responsible for summarizing ray-pretraining results
    from our proposed point cloud pretrain.
    """

    def __init__(self,
                 attn_cfgs,
                 feedforward_channels,
                 ffn_dropout=0.0,
                 operation_order=None,
                 act_cfg=dict(type='ReLU', inplace=True),
                 norm_cfg=dict(type='LN'),
                 ffn_num_fcs=2,
                 latent_render=None,
                 **kwargs):
        super().__init__(
            attn_cfgs=attn_cfgs,
            feedforward_channels=feedforward_channels,
            ffn_dropout=ffn_dropout,
            operation_order=operation_order,
            act_cfg=act_cfg,
            norm_cfg=norm_cfg,
            ffn_num_fcs=ffn_num_fcs,
            **kwargs)
        self.fp16_enabled = False

        if latent_render is not None:
            self.latent_render = ConditionalNorm(**latent_render)

    def forward(self,
                query,
                key=None,
                value=None,
                bev_pos=None,
                query_pos=None,
                key_pos=None,
                attn_masks=None,
                query_key_padding_mask=None,
                key_padding_mask=None,
                ref_2d=None,
                ref_3d=None,
                bev_h=None,
                bev_w=None,
                reference_points_cam=None,
                mask=None,
                spatial_shapes=None,
                level_start_index=None,
                prev_bev=None,
                **kwargs):
        """Forward function for `TransformerDecoderLayer`.

        **kwargs contains some specific arguments of attentions.

        Args:
            query (Tensor): The input query with shape
                [num_queries, bs, embed_dims] if
                self.batch_first is False, else
                [bs, num_queries embed_dims].
            key (Tensor): The key tensor with shape [num_keys, bs,
                embed_dims] if self.batch_first is False, else
                [bs, num_keys, embed_dims] .
            value (Tensor): The value tensor with same shape as `key`.
            query_pos (Tensor): The positional encoding for `query`.
                Default: None.
            key_pos (Tensor): The positional encoding for `key`.
                Default: None.
            attn_masks (List[Tensor] | None): 2D Tensor used in
                calculation of corresponding attention. The length of
                it should equal to the number of `attention` in
                `operation_order`. Default: None.
            query_key_padding_mask (Tensor): ByteTensor for `query`, with
                shape [bs, num_queries]. Only used in `self_attn` layer.
                Defaults to None.
            key_padding_mask (Tensor): ByteTensor for `query`, with
                shape [bs, num_keys]. Default: None.

        Returns:
            Tensor: forwarded results with shape [num_queries, bs, embed_dims].
        """

        norm_index = 0
        attn_index = 0
        ffn_index = 0
        identity = query
        if attn_masks is None:
            attn_masks = [None for _ in range(self.num_attn)]
        elif isinstance(attn_masks, torch.Tensor):
            attn_masks = [
                copy.deepcopy(attn_masks) for _ in range(self.num_attn)
            ]
            warnings.warn(f'Use same attn_mask in all attentions in '
                          f'{self.__class__.__name__} ')
        else:
            assert len(attn_masks) == self.num_attn, f'The length of ' \
                                                     f'attn_masks {len(attn_masks)} must be equal ' \
                                                     f'to the number of attention in ' \
                f'operation_order {self.num_attn}'

        for layer in self.operation_order:
            # temporal self attention
            if layer == 'self_attn':

                query = self.attentions[attn_index](
                    query,
                    prev_bev,
                    prev_bev,
                    identity if self.pre_norm else None,
                    query_pos=bev_pos,
                    key_pos=bev_pos,
                    attn_mask=attn_masks[attn_index],
                    key_padding_mask=query_key_padding_mask,
                    reference_points=ref_2d,
                    # GVAD needs the explicit token layout.  In particular,
                    # NearFar uses a non-square ``active_count x 1`` sparse
                    # sequence, whose shape cannot be inferred from length.
                    bev_h=bev_h,
                    bev_w=bev_w,
                    spatial_shapes=torch.tensor(
                        [[bev_h, bev_w]], device=query.device),
                    level_start_index=torch.tensor([0], device=query.device),
                    **kwargs)
                attn_index += 1
                identity = query

            elif layer == 'norm':
                query = self.norms[norm_index](query)
                norm_index += 1

            # spaital cross attention
            elif layer == 'cross_attn':
                query = self.attentions[attn_index](
                    query,
                    key,
                    value,
                    identity if self.pre_norm else None,
                    query_pos=query_pos,
                    key_pos=key_pos,
                    reference_points=ref_3d,
                    reference_points_cam=reference_points_cam,
                    mask=mask,
                    attn_mask=attn_masks[attn_index],
                    key_padding_mask=key_padding_mask,
                    spatial_shapes=spatial_shapes,
                    level_start_index=level_start_index,
                    **kwargs)
                attn_index += 1
                identity = query

            # unsupervised ray-wise marching operation.
            elif layer == 'latent_render':
                bs, token_num, embed_dim = query.shape
                query = self.latent_render(query.view(bs, bev_h, bev_w, embed_dim))
                query = query.view(bs, token_num, embed_dim)

            elif layer == 'ffn':
                query = self.ffns[ffn_index](
                    query, identity if self.pre_norm else None)
                ffn_index += 1

        return query
