"""Current-frame six-camera semantic + instance occupancy."""
import numpy as np
import torch
from torch.utils.checkpoint import checkpoint
from mmdet.models import DETECTORS
from mmcv.runner import auto_fp16
from .drive_occworldV2 import Drive_OccWorld_V2
from ...core.evaluation.panoptic_metrics import panoptic_counts


@DETECTORS.register_module()
class FarmSimPanoptic(Drive_OccWorld_V2):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        if self.future_pred_frame_num or self.test_future_frame_num or self.turn_on_plan:
            raise ValueError('Panoptic detector supports current-frame occupancy only')

    def obtain_ref_bev(self, img, img_metas, prev_bev, return_img_feats=False):
        if not self.training:
            return super().obtain_ref_bev(img, img_metas, prev_bev, return_img_feats)
        # Checkpoint the BEV encoder, after stochastic image augmentation. Recomputing
        # NumPy-based GridMask would otherwise draw a different mask on backward.
        features = self.extract_feat(img=img, img_metas=img_metas)
        def encode(*levels):
            return self.pts_bbox_head(list(levels), img_metas, None, only_bev=True)
        bev = checkpoint(encode, *features, use_reentrant=False)
        return (bev, features) if return_img_feats else bev

    def _forward_panoptic(self, img, img_metas):
        if img.ndim != 6 or img.shape[1] != 1:
            raise ValueError('Expected current images [B,1,N,C,H,W]')
        metas = [m[-1] for m in img_metas]
        bev = self.obtain_ref_bev(img[:, 0], metas, None)
        return self.future_pred_head.forward_panoptic(bev), metas

    @auto_fp16(apply_to=('img',))
    def forward_train(self, img=None, img_metas=None, segmentation=None, instance=None, **kwargs):
        outputs, _ = self._forward_panoptic(img, img_metas)
        b = img.shape[0]
        sem = self._format_occ_targets(segmentation, 1, b)
        if instance is None:
            raise ValueError('Panoptic training requires instance targets')
        inst = self._format_occ_targets(instance, 1, b)
        # [B,C,Z,Y,X] -> public dataset/loss [B,C,X,Y,Z].
        semantic = outputs['semantic'].float().permute(0, 1, 4, 3, 2).contiguous()
        losses = self.future_pred_head.loss_occ(semantic.unsqueeze(0), sem)
        losses.update(self.future_pred_head.instance_decoder.loss(
            outputs['instance'], sem.permute(0, 3, 2, 1), inst.permute(0, 3, 2, 1)))
        return losses

    @auto_fp16(apply_to=('img',))
    def forward_test(self, **kwargs):
        return self.simple_test(**kwargs)

    @torch.no_grad()
    def simple_test(self, img=None, img_metas=None, segmentation=None, instance=None, **kwargs):
        outputs, metas = self._forward_panoptic(img, img_metas)
        sem_zyx = outputs['semantic'].argmax(1)
        instances_zyx = self.future_pred_head.instance_decoder.predict(outputs['instance'], sem_zyx)
        pred_sem = sem_zyx.permute(0, 3, 2, 1).cpu().numpy().astype(np.uint8)
        pred_inst = instances_zyx.permute(0, 3, 2, 1).cpu().numpy().astype(np.uint32)
        target = self._format_occ_targets(segmentation, 1, img.shape[0]).cpu().numpy()
        if instance is None:
            raise ValueError('Panoptic evaluation requires instance targets')
        target_inst = self._format_occ_targets(instance, 1, img.shape[0]).cpu().numpy()
        n = self.future_pred_head.num_classes
        histogram = np.zeros((n, n), dtype=np.int64)
        counts = np.zeros((n, 4), dtype=np.float64)
        artifacts = []
        indices = torch.as_tensor(kwargs.get('sample_idx', list(range(img.shape[0])))).reshape(-1).tolist()
        include = kwargs.get('panoptic_eval_include', [True] * img.shape[0])
        for b in range(img.shape[0]):
            if not include[b]:
                continue
            valid = target[b] != 255
            histogram += np.bincount(n * target[b][valid] + pred_sem[b][valid], minlength=n*n).reshape(n, n)
            counts += panoptic_counts(pred_sem[b], pred_inst[b], target[b], target_inst[b])
            if self._return_prediction_artifacts and self._should_build_prediction_artifacts([indices[b]]):
                artifacts.append(dict(sample_index=int(indices[b]), scene_token=str(metas[b].get('scene_token', 'scene')),
                    lidar_token=str(metas[b].get('lidar_token', b)), current_pred=pred_sem[b],
                    current_gt=target[b].astype(np.uint8), current_instance_pred=pred_inst[b],
                    current_instance_gt=target_inst[b].astype(np.uint32), valid_mask=valid,
                    num_classes=n, point_cloud_range=np.asarray(self.point_cloud_range, dtype=np.float32)))
        return dict(hist_for_iou=histogram, hist_for_iou_current=histogram,
                    panoptic_counts=counts, prediction_artifacts=artifacts)
