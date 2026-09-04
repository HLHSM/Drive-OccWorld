"""FarmSim BEV attention replacements for single-frame occupancy training."""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import xavier_init
from mmcv.cnn.bricks.registry import ATTENTION
from mmcv.runner.base_module import BaseModule

from .temporal_self_attention import TemporalSelfAttention


def _as_batch_first(query, batch_first):
    return query if batch_first else query.transpose(0, 1)


def _restore_layout(tokens, batch_first):
    return tokens if batch_first else tokens.transpose(0, 1)


def _bev_shape(tokens, bev_h, bev_w):
    """Validate and infer the dense BEV grid shape."""
    token_count = tokens.shape[1]
    if bev_h is None or bev_w is None:
        side = int(round(math.sqrt(token_count)))
        if side * side != token_count:
            raise ValueError('BEV height/width are required for non-square tokens.')
        bev_h = bev_w = side
    if token_count != bev_h * bev_w:
        raise ValueError(
            f'Expected {bev_h}*{bev_w}={bev_h * bev_w} BEV tokens, got '
            f'{token_count}.')
    return int(bev_h), int(bev_w)


@ATTENTION.register_module()
class GeometryVisibleAnchorDeformableAttention(BaseModule):
    """Geometry-visible anchors plus local deformable BEV sampling.

    The local path preserves BEVFormer's query-adaptive deformable sampling.
    The anchor path follows sparse visible-to-dense reasoning: calibrated
    camera visibility and token confidence form a small set of reliable BEV
    anchors, then every query reads these anchors with a relative-coordinate
    bias.  ``use_visibility=False`` turns the anchors into ordinary average
    pooling; ``use_local_deformable=False`` produces the anchor-only ablation.
    """

    def __init__(self,
                 embed_dims=256,
                 num_heads=8,
                 num_levels=1,
                 num_points=4,
                 num_bev_queue=2,
                 anchor_grid_height=4,
                 anchor_grid_width=8,
                 dropout=0.1,
                 batch_first=True,
                 use_visibility=True,
                 use_local_deformable=True,
                 init_cfg=None,
                 **kwargs):
        super().__init__(init_cfg)
        del kwargs
        if embed_dims % num_heads:
            raise ValueError('embed_dims must be divisible by num_heads.')
        if anchor_grid_height < 1 or anchor_grid_width < 1:
            raise ValueError('anchor grid dimensions must be positive.')
        if not (use_visibility or use_local_deformable):
            raise ValueError('GVAD needs a local deformable or anchor path.')

        self.embed_dims = int(embed_dims)
        self.num_heads = int(num_heads)
        self.head_dims = self.embed_dims // self.num_heads
        self.num_levels = int(num_levels)
        self.num_points = int(num_points)
        self.num_bev_queue = int(num_bev_queue)
        self.anchor_grid_height = int(anchor_grid_height)
        self.anchor_grid_width = int(anchor_grid_width)
        self.batch_first = bool(batch_first)
        self.use_visibility = bool(use_visibility)
        self.use_local_deformable = bool(use_local_deformable)
        self.fp16_enabled = False

        if self.use_local_deformable:
            # This is the information-preserving path: it uses exactly the
            # deformable offset/weight mechanism that made the H=0 TSA strong.
            self.local_deformable = TemporalSelfAttention(
                embed_dims=self.embed_dims,
                num_heads=self.num_heads,
                num_levels=self.num_levels,
                num_points=self.num_points,
                num_bev_queue=self.num_bev_queue,
                dropout=dropout,
                batch_first=True)

        self.norm = nn.LayerNorm(self.embed_dims)
        self.anchor_confidence = nn.Linear(self.embed_dims, 1)
        self.anchor_query = nn.Linear(self.embed_dims, self.embed_dims)
        self.anchor_key = nn.Linear(self.embed_dims, self.embed_dims)
        self.anchor_value = nn.Linear(self.embed_dims, self.embed_dims)
        self.anchor_output = nn.Linear(self.embed_dims, self.embed_dims)
        self.anchor_gate = nn.Linear(self.embed_dims + 1, 1)
        # Positive coefficients impose a soft geometric locality prior, while
        # still allowing any query to read any visible anchor.
        self.distance_decay = nn.Parameter(torch.zeros(self.num_heads))
        self.visibility_logit_scale = nn.Parameter(torch.tensor(1.0))
        self.dropout = nn.Dropout(dropout)
        self.init_weights()

    def init_weights(self):
        for module in (self.anchor_confidence, self.anchor_query,
                       self.anchor_key, self.anchor_value,
                       self.anchor_output, self.anchor_gate):
            xavier_init(module, distribution='uniform', bias=0.)
        # Anchor context begins as a small, but nonzero, correction.  Unlike
        # a 1e-3 residual gate, it receives useful gradients during warm-up.
        nn.init.constant_(self.anchor_gate.bias, -1.5)

    @staticmethod
    def _query_coordinates(height, width, batch, device, dtype):
        rows = (torch.arange(height, device=device, dtype=dtype) + 0.5) / height
        cols = (torch.arange(width, device=device, dtype=dtype) + 0.5) / width
        yy, xx = torch.meshgrid(rows, cols, indexing='ij')
        return torch.stack((yy, xx), dim=-1).reshape(1, height * width, 2).expand(
            batch, -1, -1)

    @staticmethod
    def _visibility_from_mask(bev_mask, batch, tokens, device, dtype):
        """Reduce encoder projection validity ``[cam,B,HW,Z]`` to ``[B,HW]``."""
        if bev_mask is None:
            return torch.ones(batch, tokens, device=device, dtype=dtype)
        mask = bev_mask.to(device=device)
        if mask.dim() != 4:
            raise ValueError('bev_mask must have shape [camera,B,HW,Z] or [B,camera,HW,Z].')
        if mask.shape[1] == batch and mask.shape[2] == tokens:
            # Encoder-native [camera, B, HW, pillar-z].
            visibility = mask.permute(1, 2, 0, 3).float().mean(dim=(2, 3))
        elif mask.shape[0] == batch and mask.shape[2] == tokens:
            visibility = mask.float().mean(dim=(1, 3))
        else:
            raise ValueError('bev_mask dimensions do not match the BEV query.')
        return visibility.to(dtype).clamp(0.0, 1.0)

    def _pooled_anchors(self, feature, height, width):
        """Ordinary pooled anchors used by the no-visibility ablation."""
        batch = feature.shape[0]
        feature_map = feature.transpose(1, 2).reshape(
            batch, self.embed_dims, height, width)
        anchors = F.adaptive_avg_pool2d(
            feature_map, (self.anchor_grid_height, self.anchor_grid_width))
        anchors = anchors.flatten(2).transpose(1, 2)
        coordinates = self._query_coordinates(
            self.anchor_grid_height, self.anchor_grid_width, batch,
            feature.device, feature.dtype)
        valid = torch.ones(
            batch, anchors.shape[1], device=feature.device, dtype=torch.bool)
        return anchors, coordinates, valid

    def _visible_anchors(self, feature, visibility, height, width):
        """Pool one differentiable reliability-weighted anchor per BEV tile."""
        batch = feature.shape[0]
        feature_map = feature.reshape(batch, height, width, self.embed_dims)
        visibility_map = visibility.reshape(batch, height, width)
        y_coords = (torch.arange(height, device=feature.device,
                                 dtype=feature.dtype) + 0.5) / height
        x_coords = (torch.arange(width, device=feature.device,
                                 dtype=feature.dtype) + 0.5) / width
        yy, xx = torch.meshgrid(y_coords, x_coords, indexing='ij')
        coordinate_map = torch.stack((yy, xx), dim=-1)

        anchors, coordinates = [], []
        visibility_scale = self.visibility_logit_scale.clamp(-4.0, 4.0)
        for row in range(self.anchor_grid_height):
            y0 = row * height // self.anchor_grid_height
            y1 = (row + 1) * height // self.anchor_grid_height
            for col in range(self.anchor_grid_width):
                x0 = col * width // self.anchor_grid_width
                x1 = (col + 1) * width // self.anchor_grid_width
                tile = feature_map[:, y0:y1, x0:x1].reshape(batch, -1,
                                                             self.embed_dims)
                tile_visibility = visibility_map[:, y0:y1, x0:x1].reshape(
                    batch, -1)
                score = self.anchor_confidence(tile).squeeze(-1)
                score = score + visibility_scale * torch.log(
                    tile_visibility.clamp_min(1e-4))
                weight = F.softmax(score.float(), dim=-1).to(feature.dtype)
                anchors.append(torch.einsum('bk,bkc->bc', weight, tile))
                tile_coords = coordinate_map[y0:y1, x0:x1].reshape(-1, 2)
                coordinates.append(torch.einsum(
                    'bk,kd->bd', weight, tile_coords.to(feature.dtype)))
        anchors = torch.stack(anchors, dim=1)
        coordinates = torch.stack(coordinates, dim=1)
        valid = torch.ones(
            batch, anchors.shape[1], device=feature.device, dtype=torch.bool)
        return anchors, coordinates, valid

    @staticmethod
    def _sparse_query_coordinates(active_indices, dense_height, dense_width,
                                  batch, device, dtype):
        """Map NearFar's active dense-grid indices back to metric BEV coords."""
        if active_indices is None or active_indices.dim() != 1:
            raise ValueError('GVAD sparse layout requires 1-D active BEV indices.')
        active_indices = active_indices.to(device=device, dtype=torch.long)
        rows = torch.div(active_indices, dense_width, rounding_mode='floor')
        cols = active_indices.remainder(dense_width)
        coordinates = torch.stack(
            ((rows.to(dtype) + 0.5) / dense_height,
             (cols.to(dtype) + 0.5) / dense_width), dim=-1)
        return coordinates.unsqueeze(0).expand(batch, -1, -1)

    def _sparse_anchors(self, feature, visibility, query_coords):
        """Build spatial anchors from NearFar-selected tokens.

        NearFar packs nonuniformly sampled BEV cells into a 1-D sequence.  It
        is therefore invalid to pool that sequence as an ``active_count x 1``
        image.  This method bins tokens by their original dense-grid metric
        coordinates and masks empty bins, retaining GVAD's geometric meaning.
        """
        batch, tokens, _ = feature.shape
        if query_coords.shape[:2] != (batch, tokens):
            raise ValueError('GVAD sparse coordinates do not match active tokens.')
        row_bin = torch.clamp(
            (query_coords[0, :, 0] * self.anchor_grid_height).long(),
            max=self.anchor_grid_height - 1)
        col_bin = torch.clamp(
            (query_coords[0, :, 1] * self.anchor_grid_width).long(),
            max=self.anchor_grid_width - 1)
        visibility_scale = self.visibility_logit_scale.clamp(-4.0, 4.0)
        anchors, coordinates, valid = [], [], []
        for row in range(self.anchor_grid_height):
            for col in range(self.anchor_grid_width):
                in_tile = (row_bin == row) & (col_bin == col)
                tile_center = query_coords.new_tensor(
                    ((row + 0.5) / self.anchor_grid_height,
                     (col + 0.5) / self.anchor_grid_width))
                if not bool(in_tile.any()):
                    anchors.append(feature.new_zeros(batch, self.embed_dims))
                    coordinates.append(tile_center.unsqueeze(0).expand(batch, -1))
                    valid.append(torch.zeros(batch, device=feature.device,
                                             dtype=torch.bool))
                    continue
                tile = feature[:, in_tile, :]
                tile_coords = query_coords[:, in_tile, :]
                if self.use_visibility:
                    score = self.anchor_confidence(tile).squeeze(-1)
                    score = score + visibility_scale * torch.log(
                        visibility[:, in_tile].clamp_min(1e-4))
                    weight = F.softmax(score.float(), dim=-1).to(feature.dtype)
                    anchors.append(torch.einsum('bk,bkc->bc', weight, tile))
                    coordinates.append(torch.einsum('bk,bkd->bd', weight,
                                                     tile_coords))
                else:
                    anchors.append(tile.mean(dim=1))
                    coordinates.append(tile_coords.mean(dim=1))
                valid.append(torch.ones(batch, device=feature.device,
                                        dtype=torch.bool))
        return (torch.stack(anchors, dim=1), torch.stack(coordinates, dim=1),
                torch.stack(valid, dim=1))

    def _anchor_context(self, feature, visibility, height, width,
                        query_coords=None, sparse_layout=False):
        if sparse_layout:
            anchors, anchor_coords, anchor_valid = self._sparse_anchors(
                feature, visibility, query_coords)
        elif self.use_visibility:
            anchors, anchor_coords, anchor_valid = self._visible_anchors(
                feature, visibility, height, width)
        else:
            anchors, anchor_coords, anchor_valid = self._pooled_anchors(
                feature, height, width)

        batch, tokens, _ = feature.shape
        query = self.anchor_query(feature).reshape(
            batch, tokens, self.num_heads, self.head_dims).transpose(1, 2)
        key = self.anchor_key(anchors).reshape(
            batch, -1, self.num_heads, self.head_dims).transpose(1, 2)
        value = self.anchor_value(anchors).reshape(
            batch, -1, self.num_heads, self.head_dims).transpose(1, 2)
        attention = torch.matmul(query, key.transpose(-1, -2))
        attention = attention * (self.head_dims ** -0.5)
        if query_coords is None:
            query_coords = self._query_coordinates(
                height, width, batch, feature.device, feature.dtype)
        squared_distance = (query_coords[:, :, None, :] -
                            anchor_coords[:, None, :, :]).square().sum(dim=-1)
        attention = attention - F.softplus(self.distance_decay).view(
            1, self.num_heads, 1, 1) * squared_distance.unsqueeze(1)
        attention = attention.masked_fill(
            ~anchor_valid[:, None, None, :], -1e4)
        attention = F.softmax(attention.float(), dim=-1).to(query.dtype)
        context = torch.matmul(attention, value).transpose(1, 2).reshape(
            batch, tokens, self.embed_dims)
        return self.anchor_output(context)

    def forward(self,
                query,
                key=None,
                value=None,
                identity=None,
                query_pos=None,
                **kwargs):
        del key, value
        query = _as_batch_first(query, self.batch_first)
        identity = query if identity is None else _as_batch_first(
            identity, self.batch_first)
        height, width = _bev_shape(query, kwargs.get('bev_h'), kwargs.get('bev_w'))
        sparse_layout = bool(kwargs.get('gvad_sparse_layout', False))
        position = None if query_pos is None else _as_batch_first(
            query_pos, self.batch_first)
        feature = self.norm(query if position is None else query + position)
        visibility = self._visibility_from_mask(
            kwargs.get('bev_mask'), query.shape[0], query.shape[1],
            query.device, feature.dtype)

        if sparse_layout:
            dense_height = kwargs.get('gvad_dense_bev_h')
            dense_width = kwargs.get('gvad_dense_bev_w')
            if dense_height is None or dense_width is None:
                raise ValueError('GVAD sparse layout requires dense BEV dimensions.')
            query_coords = self._sparse_query_coordinates(
                kwargs.get('gvad_active_indices'), int(dense_height),
                int(dense_width), query.shape[0], query.device, feature.dtype)
        else:
            query_coords = None

        if self.use_local_deformable and not sparse_layout:
            local_output = self.local_deformable(
                query, identity=None, query_pos=position, **kwargs)
            local_delta = local_output - query
        else:
            local_delta = query.new_zeros(query.shape)

        anchor_delta = self._anchor_context(
            feature, visibility, height, width, query_coords=query_coords,
            sparse_layout=sparse_layout)
        # Trust global anchors most where the geometric projection offers less
        # direct evidence, but retain a nonzero correction in visible regions.
        anchor_gate = torch.sigmoid(self.anchor_gate(torch.cat(
            (feature, visibility.unsqueeze(-1)), dim=-1)))
        anchor_gate = anchor_gate * (1.0 - 0.75 * visibility.unsqueeze(-1))
        output = identity + self.dropout(local_delta + anchor_gate * anchor_delta)
        return _restore_layout(output, self.batch_first)


@ATTENTION.register_module()
class DirectionalDecaySelectiveRetention(BaseModule):
    """Distance-decayed four-direction retention with selective local kernels."""

    def __init__(self,
                 embed_dims=256,
                 retention_radius=15,
                 local_dilation=3,
                 dropout=0.1,
                 batch_first=True,
                 init_cfg=None,
                 **kwargs):
        super().__init__(init_cfg)
        del kwargs
        if retention_radius < 2:
            raise ValueError('retention_radius must be at least 2.')
        if local_dilation < 1:
            raise ValueError('local_dilation must be positive.')
        self.embed_dims = int(embed_dims)
        self.retention_radius = int(retention_radius)
        self.local_dilation = int(local_dilation)
        self.batch_first = bool(batch_first)
        self.fp16_enabled = False

        self.norm = nn.LayerNorm(self.embed_dims)
        self.direction_logits = nn.Parameter(torch.zeros(4, self.embed_dims))
        self.local_short = nn.Conv2d(self.embed_dims, self.embed_dims, 3,
                                     padding=1, groups=self.embed_dims,
                                     bias=False)
        self.local_long = nn.Conv2d(
            self.embed_dims, self.embed_dims, 7,
            padding=3 * self.local_dilation, dilation=self.local_dilation,
            groups=self.embed_dims, bias=False)
        self.kernel_selector = nn.Conv2d(2, 2, 3, padding=1)
        self.variance_gate = nn.Conv2d(1, 1, 1)
        self.output_proj = nn.Conv2d(self.embed_dims, self.embed_dims, 1)
        self.gamma = nn.Parameter(torch.full((1,), 1e-3))
        self.dropout = nn.Dropout(dropout)
        self.init_weights()

    def init_weights(self):
        for module in (self.local_short, self.local_long):
            nn.init.kaiming_normal_(module.weight, mode='fan_out',
                                    nonlinearity='relu')
        for module in (self.kernel_selector, self.variance_gate,
                       self.output_proj):
            xavier_init(module, distribution='uniform', bias=0.)

    def _directional_kernel(self, alpha, axis, reverse, dtype):
        radius = self.retention_radius
        offsets = torch.arange(-radius, radius + 1, device=alpha.device,
                               dtype=dtype)
        distance = (-offsets if reverse else offsets).clamp_min(0)
        weights = alpha.to(dtype).unsqueeze(-1).pow(distance.unsqueeze(0))
        weights = weights / weights.sum(dim=-1, keepdim=True).clamp_min(1e-6)
        return weights[:, None, None, :] if axis == 'horizontal' else \
            weights[:, None, :, None]

    def _retention(self, feature):
        alphas = 0.5 + 0.45 * torch.sigmoid(self.direction_logits)
        radius = self.retention_radius
        left = F.conv2d(feature, self._directional_kernel(
            alphas[0], 'horizontal', True, feature.dtype),
            padding=(0, radius), groups=self.embed_dims)
        right = F.conv2d(feature, self._directional_kernel(
            alphas[1], 'horizontal', False, feature.dtype),
            padding=(0, radius), groups=self.embed_dims)
        up = F.conv2d(feature, self._directional_kernel(
            alphas[2], 'vertical', True, feature.dtype),
            padding=(radius, 0), groups=self.embed_dims)
        down = F.conv2d(feature, self._directional_kernel(
            alphas[3], 'vertical', False, feature.dtype),
            padding=(radius, 0), groups=self.embed_dims)
        return 0.25 * (left + right + up + down)

    def forward(self, query, key=None, value=None, identity=None,
                query_pos=None, **kwargs):
        del key, value
        query = _as_batch_first(query, self.batch_first)
        identity = query if identity is None else _as_batch_first(
            identity, self.batch_first)
        height, width = _bev_shape(query, kwargs.get('bev_h'), kwargs.get('bev_w'))
        source = query if query_pos is None else query + _as_batch_first(
            query_pos, self.batch_first)
        feature = self.norm(source).transpose(1, 2).reshape(
            query.shape[0], self.embed_dims, height, width)
        retained = self._retention(feature)
        short = self.local_short(feature)
        long = self.local_long(feature)
        descriptors = torch.cat(((short + long).mean(dim=1, keepdim=True),
                                 (short + long).amax(dim=1, keepdim=True)), dim=1)
        local_weights = F.softmax(self.kernel_selector(descriptors), dim=1)
        local = local_weights[:, :1] * short + local_weights[:, 1:] * long
        local_mean = F.avg_pool2d(feature, 5, stride=1, padding=2)
        variance = (feature - local_mean).square().mean(dim=1, keepdim=True)
        detail_route = torch.sigmoid(self.variance_gate(variance))
        mixed = detail_route * local + (1.0 - detail_route) * retained
        mixed = self.output_proj(mixed).flatten(2).transpose(1, 2)
        return _restore_layout(
            identity + self.gamma * self.dropout(mixed), self.batch_first)
