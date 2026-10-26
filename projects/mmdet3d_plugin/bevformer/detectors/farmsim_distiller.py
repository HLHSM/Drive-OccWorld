"""Training-only R101-to-R50 distillation for current-frame FarmSim occupancy."""

import copy

import torch
from torch import nn
from torch.nn import functional as F
from mmcv import Config
from mmcv.runner import auto_fp16, load_checkpoint, wrap_fp16_model
from mmdet.models import DETECTORS
from mmdet3d.models import build_model

from .drive_occworldV2 import Drive_OccWorld_V2


def balanced_occupancy_kl(student, teacher, target, temperature=2.0):
    """Mean KL per GT class, using only valid, correctly predicted teacher voxels.

    Logits are [B,C,X,Y,Z]. Absent classes and an entirely invalid mask return
    differentiable zeros. FP32 reductions keep AMP temperature/softmax stable.
    """
    if (student.shape != teacher.shape or
            student.shape[:1] + student.shape[2:] != target.shape):
        raise ValueError('Distillation logits and GT must share the same voxel grid.')
    teacher = teacher.detach().float()
    student = student.float()
    valid = ((target != 255) & (target >= 0) &
             (target < student.shape[1]))
    reliable = valid & (teacher.argmax(dim=1) == target)
    divergence = F.kl_div(
        F.log_softmax(student / temperature, dim=1),
        F.softmax(teacher / temperature, dim=1), reduction='none').sum(dim=1)
    class_losses = []
    for class_id in range(student.shape[1]):
        mask = reliable & (target == class_id)
        if mask.any():
            class_losses.append(divergence[mask].mean())
    loss = (torch.stack(class_losses).mean() if class_losses
            else student.sum() * 0.0)
    fraction = reliable.sum().float() / valid.sum().clamp_min(1)
    return loss * temperature**2, fraction


def normalized_spatial_features(feature):
    """Remove per-channel offsets/scales before matching different backbones."""
    feature = feature.float()
    mean = feature.mean(dim=(-2, -1), keepdim=True)
    variance = feature.var(dim=(-2, -1), keepdim=True, unbiased=False)
    return (feature - mean) * variance.clamp_min(1e-6).rsqrt()


def channel_spatial_kl(student, teacher, temperature=1.0):
    """Scale-normalized CWD over H*W for each image and feature channel."""
    if student.shape != teacher.shape:
        raise ValueError('FPN distillation requires matching feature shapes.')
    student = normalized_spatial_features(student).flatten(start_dim=2)
    teacher = normalized_spatial_features(teacher.detach()).flatten(start_dim=2)
    return F.kl_div(
        F.log_softmax(student / temperature, dim=-1),
        F.softmax(teacher / temperature, dim=-1),
        reduction='none').sum(dim=-1).mean() * temperature**2


@DETECTORS.register_module()
class FarmSimOccupancyDistiller(Drive_OccWorld_V2):
    """An unchanged inference student with optional training distillation.

    The teacher is deliberately NOT registered as a child module: it must not
    enter the optimizer, DDP synchronization, EMA copies or student checkpoints.
    It is constructed lazily only during a training forward on the local GPU.
    FPN adapters are student training parameters and remain resumable in training
    checkpoints, but are never evaluated by the inference path.
    """

    def __init__(self, occupancy_distillation, **kwargs):
        super().__init__(**kwargs)
        self.kd_cfg = copy.deepcopy(occupancy_distillation)
        self.kd_logit_weight = float(self.kd_cfg.get('logit_weight', 0.0))
        self.kd_fpn_weight = float(self.kd_cfg.get('fpn_weight', 0.0))
        self.kd_levels = tuple(self.kd_cfg.get('fpn_levels', (0, 1)))
        if not self.only_train_cur_frame or self.future_pred_head.history_queue_length != 0:
            raise ValueError('FarmSim distillation requires current-frame, zero-history occupancy.')
        if self.turn_on_plan or self.use_lwm:
            raise ValueError('FarmSim distillation is for occupancy-only training.')
        if self.grid_mask_backbone_feat or self.grid_mask_fpn_feat:
            raise ValueError('Distillation supports shared image GridMask only, not feature GridMask.')
        if self.random_drop_image_rate or self.random_drop_prev_rate:
            raise ValueError('Disable random frame dropping when distilling aligned images.')
        if min(self.kd_logit_weight, self.kd_fpn_weight) < 0:
            raise ValueError('Distillation loss weights must be non-negative.')
        if max(self.kd_logit_weight, self.kd_fpn_weight) <= 0:
            raise ValueError('Use Drive_OccWorld_V2 for the no-distillation control.')
        if min(float(self.kd_cfg.get('logit_temperature', 2.0)),
               float(self.kd_cfg.get('fpn_temperature', 1.0))) <= 0:
            raise ValueError('Distillation temperatures must be positive.')
        if float(self.kd_cfg.get('warmup_epochs', 0.5)) < 0:
            raise ValueError('Distillation warm-up must be non-negative.')
        if not self.kd_levels or min(self.kd_levels) < 0:
            raise ValueError('Distillation FPN levels must be non-negative and nonempty.')
        self.kd_fpn_adapters = nn.ModuleList()
        if self.kd_fpn_weight > 0:
            channels = self.img_neck.out_channels
            if max(self.kd_levels) >= self.img_neck.num_outs:
                raise ValueError('Distillation FPN level exceeds the student FPN pyramid.')
            with torch.random.fork_rng(devices=[]):
                for _ in self.kd_levels:
                    adapter = nn.Conv2d(channels, channels, 1, bias=False)
                    nn.init.eye_(adapter.weight[:, :, 0, 0])
                    self.kd_fpn_adapters.append(adapter)
        object.__setattr__(self, '_kd_teacher', None)
        self._kd_capture = False
        self._kd_progress = 0.0
        self._clear_kd_tensors()

    def _clear_kd_tensors(self):
        self._kd_teacher_features = None
        self._kd_student_features = None
        self._kd_teacher_logits = None
        self._kd_student_logits = None

    def set_distillation_progress(self, epoch_fraction):
        self._kd_progress = float(epoch_fraction)

    def _get_teacher(self, device):
        teacher = self._kd_teacher
        if teacher is None:
            cfg = Config.fromfile(self.kd_cfg['teacher_config'])
            model_cfg = copy.deepcopy(cfg.model)
            if model_cfg['type'] != 'Drive_OccWorld_V2':
                raise ValueError('Teacher config must describe a plain Drive_OccWorld_V2.')
            expected = (self.bev_h, self.bev_w,
                        self.future_pred_head.num_pred_height,
                        self.future_pred_head.num_classes)
            actual = (model_cfg['bev_h'], model_cfg['bev_w'],
                      model_cfg['future_pred_head']['num_pred_height'],
                      model_cfg['future_pred_head']['num_classes'])
            if (actual != expected or
                    tuple(model_cfg['point_cloud_range']) != tuple(self.point_cloud_range)):
                raise ValueError('Teacher and student occupancy geometry/classes must match.')
            if (model_cfg['img_backbone']['depth'] != 101 or
                    model_cfg['future_pred_frame_num'] != 0 or
                    model_cfg['future_pred_head']['history_queue_length'] != 0):
                raise ValueError('Teacher must be an R101 zero-history current-occupancy model.')
            model_cfg.update(grid_mask_image=False, grid_mask_backbone_feat=False,
                             grid_mask_fpn_feat=False, train_cfg=None)
            # Constructing a teacher must not perturb the student's RNG stream.
            devices = [device.index] if device.type == 'cuda' else []
            with torch.random.fork_rng(devices=devices):
                teacher = build_model(model_cfg, test_cfg=cfg.get('test_cfg'))
                # The normal training init removes this unused detection-only
                # layer. Match that checkpoint contract without reinitializing
                # all teacher modules before loading their trained weights.
                if hasattr(teacher.pts_bbox_head.transformer, 'reference_points'):
                    del teacher.pts_bbox_head.transformer.reference_points
                load_checkpoint(teacher, self.kd_cfg['teacher_checkpoint'],
                                map_location='cpu', strict=True)
                teacher.requires_grad_(False)
                teacher.to(device)
                if self.fp16_enabled:
                    wrap_fp16_model(teacher)
                teacher.eval()
            object.__setattr__(self, '_kd_teacher', teacher)
        return teacher

    @auto_fp16(apply_to=('img',))
    def extract_feat(self, img, img_metas=None, len_queue=None):
        if not self._kd_capture:
            return super().extract_feat(img, img_metas=img_metas, len_queue=len_queue)
        if img.ndim != 5 or len_queue is not None:
            raise ValueError('Distillation expects one current [B,N,C,H,W] image batch.')
        # Apply exactly one mask, at the same point in the training forward as
        # the control. Both networks see the identical augmented pixels.
        shared = img
        if self.use_grid_mask and self.grid_mask_image:
            shared = self.grid_mask(img.reshape(-1, *img.shape[-3:])).reshape_as(img)
        teacher = self._get_teacher(img.device)
        teacher_metas = copy.deepcopy(img_metas)
        for meta in teacher_metas:
            meta['prev_bev_exists'] = False
        with torch.no_grad():
            if self.kd_logit_weight > 0:
                bev, features = teacher.obtain_ref_bev(
                    shared, teacher_metas, None, return_img_feats=True)
                predictions = teacher.current_occ_prediction(bev)['next_bev_preds']
                self._kd_teacher_logits = self._final_logits(predictions).detach()
            else:
                features = teacher.extract_feat(shared, img_metas=teacher_metas)
            if self.kd_fpn_weight > 0:
                self._kd_teacher_features = [features[level].detach() for level in self.kd_levels]
        mask_image = self.grid_mask_image
        self.grid_mask_image = False
        try:
            features = super().extract_feat(shared, img_metas=img_metas)
        finally:
            self.grid_mask_image = mask_image
        if self.kd_fpn_weight > 0:
            self._kd_student_features = [features[level] for level in self.kd_levels]
        return features

    def _final_logits(self, predictions):
        # [T,aux,B,HW,Z,C] -> [B,C,X,Y,Z], exactly matching compute_occ_loss.
        logits = predictions[0, -1]
        batch, tokens, depth, classes = logits.shape
        height, width = self._occupancy_prediction_hw(tokens)
        return logits.permute(0, 3, 1, 2).reshape(
            batch, classes, width, height, depth).transpose(2, 3)

    def current_occ_prediction(self, *args, **kwargs):
        output = super().current_occ_prediction(*args, **kwargs)
        if self._kd_capture and self.kd_logit_weight > 0:
            self._kd_student_logits = self._final_logits(output['next_bev_preds'])
        return output

    @auto_fp16(apply_to=('img',))
    def forward_train(self, img=None, img_metas=None, segmentation=None, **kwargs):
        if img.shape[1] != 1:
            raise ValueError('Distillation requires exactly one current frame per sample.')
        self._kd_capture = True
        try:
            losses = super().forward_train(
                img=img, img_metas=img_metas, segmentation=segmentation, **kwargs)
            warmup = float(self.kd_cfg.get('warmup_epochs', 0.5))
            scale = min(1.0, self._kd_progress / warmup) if warmup > 0 else 1.0
            if self.kd_logit_weight > 0:
                target = self._format_occ_targets(segmentation, 1, img.shape[0])
                loss, reliable_fraction = balanced_occupancy_kl(
                    self._kd_student_logits, self._kd_teacher_logits, target,
                    float(self.kd_cfg.get('logit_temperature', 2.0)))
                losses['loss_kd_logits'] = self.kd_logit_weight * scale * loss
                losses['kd_reliable_fraction'] = reliable_fraction
            if self.kd_fpn_weight > 0:
                feature_losses = []
                for adapter, student, teacher in zip(
                        self.kd_fpn_adapters, self._kd_student_features,
                        self._kd_teacher_features):
                    # Raw pretrained FPN amplitudes differ substantially.
                    # Normalize adapter inputs as well; run these small KD-only
                    # projections in FP32 to avoid AMP gradient overflow.
                    with torch.autocast(device_type=student.device.type, enabled=False):
                        student = adapter(normalized_spatial_features(
                            student.flatten(0, 1)))
                    feature_losses.append(channel_spatial_kl(
                        student, teacher.flatten(0, 1),
                        float(self.kd_cfg.get('fpn_temperature', 1.0))))
                losses['loss_kd_fpn'] = (self.kd_fpn_weight * scale *
                                         torch.stack(feature_losses).mean())
            losses['kd_warmup_scale'] = img.new_tensor(scale, dtype=torch.float32)
            return losses
        finally:
            self._kd_capture = False
            self._clear_kd_tensors()
