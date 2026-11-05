"""Shared AMoE volume with sampled set-mask supervision (no dense Q-volume in training)."""
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint
from mmdet.models import HEADS
from mmcv.runner import force_fp32
from .world_head_v1 import WorldHeadV1


def linear_sum_assignment(cost):
    """Rectangular Hungarian assignment, rows=queries >= columns=GT; no SciPy dependency."""
    import numpy as np
    a = np.asarray(cost, dtype=np.float64).T
    n, m = a.shape
    if n > m or not np.isfinite(a).all():
        raise ValueError('Assignment requires finite costs and queries >= targets')
    u, v = np.zeros(n + 1), np.zeros(m + 1)
    p, way = np.zeros(m + 1, dtype=int), np.zeros(m + 1, dtype=int)
    for i in range(1, n + 1):
        p[0] = i
        j0 = 0
        minimum = np.full(m + 1, np.inf)
        used = np.zeros(m + 1, dtype=bool)
        while True:
            used[j0] = True
            i0 = p[j0]
            for j in range(1, m + 1):
                if not used[j]:
                    cur = a[i0 - 1, j - 1] - u[i0] - v[j]
                    if cur < minimum[j]:
                        minimum[j], way[j] = cur, j0
            candidates = np.where(~used[1:])[0] + 1
            j1 = candidates[np.argmin(minimum[candidates])]
            delta = minimum[j1]
            u[p[used]] += delta
            v[used] -= delta
            minimum[~used] -= delta
            j0 = j1
            if p[j0] == 0:
                break
        while True:
            j1 = way[j0]
            p[j0] = p[j1]
            j0 = j1
            if j0 == 0:
                break
    queries = np.where(p[1:] > 0)[0]
    return queries, p[queries + 1] - 1


class PanopticMaskDecoder(nn.Module):
    def __init__(self, in_channels, hidden=64, num_queries=100, num_layers=3,
                 thing_ids=(4, 5, 7, 8, 9, 10), memory_shape=(5, 25, 25),
                 sample_points=4096, points_per_instance=16, score_threshold=.3):
        super().__init__()
        if hidden % 8 or num_queries < 1 or sample_points < 1 or points_per_instance < 1:
            raise ValueError('Invalid mask decoder dimensions/sampling')
        self.fp16_enabled = False
        self.thing_ids = tuple(thing_ids)
        self.num_queries = num_queries
        self.memory_shape = tuple(memory_shape)
        self.sample_points = sample_points
        self.points_per_instance = points_per_instance
        self.score_threshold = score_threshold
        self.mask_features = nn.Conv3d(in_channels, hidden, 1)
        self.position = nn.Linear(3, hidden)
        self.queries = nn.Embedding(num_queries, hidden)
        layer = nn.TransformerDecoderLayer(hidden, 8, hidden * 4, dropout=0.)
        self.decoder = nn.TransformerDecoder(layer, num_layers)
        self.classifier = nn.Linear(hidden, len(thing_ids) + 1)
        self.mask_embed = nn.Sequential(nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, hidden))
        weights = torch.ones(len(thing_ids) + 1)
        weights[-1] = .1
        self.register_buffer('class_weights', weights)

    def forward(self, volume):
        features = self.mask_features(volume)
        memory = F.adaptive_avg_pool3d(features, self.memory_shape)
        axes = [torch.linspace(-1, 1, n, device=volume.device, dtype=memory.dtype)
                for n in memory.shape[-3:]]
        # Explicit [Z,Y,X] positional coordinates, in the same order as flattening.
        coords = torch.stack(torch.meshgrid(*axes, indexing='ij'), -1).reshape(-1, 3)
        memory = memory.flatten(2).permute(2, 0, 1) + self.position(coords).unsqueeze(1)
        query = self.queries.weight.unsqueeze(1).expand(-1, volume.shape[0], -1)
        decoded = self.decoder(query, memory).transpose(0, 1)
        return dict(classes=self.classifier(decoded), embeddings=self.mask_embed(decoded), features=features)

    @force_fp32(apply_to=('outputs',))
    def loss(self, outputs, semantics, instances):
        # All three arrays flatten [Z,Y,X]. Matching samples always include each GT instance.
        features = outputs['features'].float().flatten(2)
        embeddings = outputs['embeddings'].float()
        classes = outputs['classes'].float()
        class_loss = classes.sum() * 0
        mask_loss = (features.sum() + embeddings.sum()) * 0
        dice_loss = mask_loss
        for b in range(classes.shape[0]):
            sem, ids = semantics[b].flatten(), instances[b].flatten()
            valid_indices = torch.nonzero(sem != 255, as_tuple=False).flatten()
            if not valid_indices.numel():
                class_loss = class_loss + classes[b].sum() * 0
                continue
            keys, labels, groups = [], [], []
            for c in self.thing_ids:
                for identity in ids[sem == c].unique():
                    if identity.item() in (0, 4294967295):
                        raise ValueError('Thing target has missing instance identity')
                    group = torch.nonzero((sem == c) & (ids == identity), as_tuple=False).flatten()
                    groups.append(group)
                    keys.append(identity)
                    labels.append(self.thing_ids.index(c))
            if len(groups) > self.num_queries:
                raise ValueError(f'{len(groups)} GT instances exceed {self.num_queries} queries; increase num_queries')
            # Uniform valid background + per-instance samples avoid dropping tiny objects.
            count = min(self.sample_points, valid_indices.numel())
            selected = [valid_indices[torch.randint(valid_indices.numel(), (count,), device=sem.device)]]
            for group in groups:
                selected.append(group[torch.randint(group.numel(), (self.points_per_instance,), device=sem.device)])
            selected = torch.cat(selected)
            logits = embeddings[b] @ features[b, :, selected]
            class_targets = torch.full((self.num_queries,), len(self.thing_ids), device=sem.device, dtype=torch.long)
            if groups:
                targets = torch.stack([((sem[selected] == self.thing_ids[label]) &
                                        (ids[selected] == identity)).float()
                                       for identity, label in zip(keys, labels)])
                label_tensor = torch.tensor(labels, device=sem.device)
                with torch.no_grad():
                    # Pairwise BCE without expanding [Q,GT,P].
                    p = logits.sigmoid()
                    bce = (F.softplus(logits).mean(-1, keepdim=True)
                           - logits @ targets.T / targets.shape[-1])
                    dice = 1 - (2 * p @ targets.T + 1) / (p.sum(-1, keepdim=True) + targets.sum(-1)[None] + 1)
                    cost = -classes[b].softmax(-1)[:, label_tensor] + 2 * bce + 2 * dice
                    qi, gi = linear_sum_assignment(cost.cpu().numpy())
                    qi = torch.as_tensor(qi, device=sem.device)
                    gi = torch.as_tensor(gi, device=sem.device)
                class_targets[qi] = label_tensor[gi]
                matched, truth = logits[qi], targets[gi]
                mask_loss = mask_loss + F.binary_cross_entropy_with_logits(matched, truth)
                p = matched.sigmoid()
                dice_loss = dice_loss + (1 - (2 * (p * truth).sum(-1) + 1) /
                                          (p.sum(-1) + truth.sum(-1) + 1)).mean()
            class_loss = class_loss + F.cross_entropy(classes[b], class_targets, weight=self.class_weights.float())
        n = classes.shape[0]
        return dict(loss_panoptic_cls=class_loss / n, loss_panoptic_mask=2 * mask_loss / n,
                    loss_panoptic_dice=2 * dice_loss / n)

    @torch.no_grad()
    @force_fp32(apply_to=('outputs',))
    def predict(self, outputs, semantics, chunk_size=65536):
        """Class-consistent masks; rejected thing voxels keep ID0 and count in PQ."""
        features = outputs['features'].float().flatten(2)
        embeddings = outputs['embeddings'].float()
        probabilities = outputs['classes'].float().softmax(-1)
        confidence, labels = probabilities.max(-1)
        result = torch.zeros_like(semantics, dtype=torch.long)
        for b in range(semantics.shape[0]):
            flat_sem, flat_result = semantics[b].flatten(), result[b].view(-1)
            for ci, c in enumerate(self.thing_ids):
                queries = torch.nonzero((labels[b] == ci) & (confidence[b] >= self.score_threshold),
                                        as_tuple=False).flatten()
                positions = torch.nonzero(flat_sem == c, as_tuple=False).flatten()
                if not queries.numel():
                    continue
                for positions_chunk in positions.split(chunk_size):
                    p = (embeddings[b, queries] @ features[b, :, positions_chunk]).sigmoid()
                    score, winner = (p * confidence[b, queries, None]).max(0)
                    accepted = p.gather(0, winner[None]).squeeze(0) >= .5
                    flat_result[positions_chunk[accepted]] = queries[winner[accepted]] + 1
        return result


@HEADS.register_module()
class PanopticWorldHead(WorldHeadV1):
    def __init__(self, instance_decoder=None, **kwargs):
        super().__init__(**kwargs)
        if not self.use_agri_amoe_decoder or self.history_queue_length != 0:
            raise ValueError('PanopticWorldHead requires AMoE and zero history')
        self.instance_decoder = PanopticMaskDecoder(self.agri_amoe_channels, **(instance_decoder or {}))

    def forward_panoptic(self, bev):
        b, tokens, _ = bev.shape
        if tokens != self.bev_h * self.bev_w:
            raise ValueError('BEV grid size mismatch')
        # BEVFormer flattening is y-major, x-fastest. Internal volume is [Z,Y,X].
        volume = (checkpoint(self._shared_volume, bev, use_reentrant=False)
                  if self.training else self._shared_volume(bev))
        return dict(semantic=self.agri_amoe_out(volume), instance=self.instance_decoder(volume))

    def _shared_volume(self, bev):
        volume = self.agri_amoe_lift(bev).view(bev.shape[0], self.bev_h, self.bev_w,
                    self.agri_amoe_channels, self.num_pred_height).permute(0, 3, 4, 1, 2)
        return self.agri_amoe(volume)
