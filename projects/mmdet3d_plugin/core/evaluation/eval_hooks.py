
# Note: Considering that MMCV's EvalHook updated its interface in V1.3.16,
# in order to avoid strong version dependency, we did not directly
# inherit EvalHook but BaseDistEvalHook.

import bisect
import os.path as osp

import mmcv
import numpy as np
import torch.distributed as dist
from mmcv.runner import DistEvalHook as BaseDistEvalHook
from mmcv.runner import EvalHook as BaseEvalHook
from torch.nn.modules.batchnorm import _BatchNorm
from mmdet.core.evaluation.eval_hooks import DistEvalHook


_OCCUPANCY_METRIC_PREFIXES = {
    'hist_for_iou': 'occ_all',
    'hist_for_iou_current': 'occ_current',
    'hist_for_iou_future': 'occ_future',
    'hist_for_iou_future_time_weighting': 'occ_future_time_weighted',
}


def _occupancy_class_names(class_names, num_classes):
    """Return stable, log-safe class names matching a confusion matrix."""
    if class_names is None or len(class_names) != num_classes:
        return tuple(f'class_{index}' for index in range(num_classes))
    return tuple(str(name).replace('/', '_').replace(' ', '_')
                 for name in class_names)


def _as_aggregated_confusion_matrix(value):
    """Collapse per-rank confusion matrices to one ``[C, C]`` matrix.

    ``collect_results_cpu`` returns one already-summed matrix for each rank.
    It therefore produces ``[world_size, C, C]`` in distributed validation,
    rather than the historical single ``[1, C, C]`` special case.
    """
    hist = np.asarray(value)
    if hist.ndim < 2 or hist.shape[-1] != hist.shape[-2]:
        return None
    if hist.ndim == 2:
        return hist
    return hist.reshape(-1, hist.shape[-2], hist.shape[-1]).sum(axis=0)


def _with_occupancy_summaries(results, class_names=None):
    """Convert occupancy confusion matrices into compact scalar log metrics."""
    if not isinstance(results, dict):
        return results, []

    ordered = {}
    summary = []
    for key, value in results.items():
        prefix = _OCCUPANCY_METRIC_PREFIXES.get(key)
        if prefix is None:
            # Preserve non-occupancy results (e.g. planning metrics).
            ordered[key] = value
            continue

        hist = _as_aggregated_confusion_matrix(value)
        if hist is None:
            # Future occupancy is disabled in current-only experiments.  Make
            # that explicit instead of logging a misleading zero mIoU.
            ordered[f'{prefix}_available'] = 0.0
            if prefix.startswith('occ_future'):
                summary.append(f'{prefix}: unavailable')
            continue

        hist = hist.astype(np.float64, copy=False)
        diagonal = np.diag(hist)
        union = hist.sum(axis=1) + hist.sum(axis=0) - diagonal
        present = union > 0
        iou = np.divide(diagonal, union, out=np.zeros_like(diagonal),
                        where=present)
        names = _occupancy_class_names(class_names, len(iou))

        ordered[f'{prefix}_available'] = 1.0
        ordered[f'{prefix}_mIoU'] = (
            float(iou[present].mean()) if present.any() else 0.0)
        ordered[f'{prefix}_mIoU_all'] = float(iou.mean()) if len(iou) else 0.0
        total = float(hist.sum())
        ordered[f'{prefix}_voxel_acc'] = (
            float(diagonal.sum() / total) if total else 0.0)
        for class_name, class_iou in zip(names, iou):
            ordered[f'{prefix}_IoU_{class_name}'] = float(class_iou)

        per_class = ', '.join(
            f'{class_name}={class_iou:.4f}'
            for class_name, class_iou in zip(names, iou))
        summary.append(
            f'{prefix}: mIoU={ordered[f"{prefix}_mIoU"]:.4f}, '
            f'mIoU_all={ordered[f"{prefix}_mIoU_all"]:.4f}, '
            f'voxel_acc={ordered[f"{prefix}_voxel_acc"]:.4f}; '
            f'IoU[{per_class}]')
    return ordered, summary


def _with_planning_summaries(results):
    """Flatten stateful trajectory metrics into JSON-safe epoch scalars."""
    if not isinstance(results, dict):
        return results, []

    results = dict(results)
    planning_results = results.pop('planning_results_computed', None)
    if not planning_results:
        return results, []

    summary = []
    for metric_name, values in planning_results.items():
        if hasattr(values, 'detach'):
            values = values.detach().cpu().tolist()
        for index, value in enumerate(values):
            horizon = (index + 1) * 0.5
            key = f'planning_{metric_name}_{horizon:.1f}s'
            results[key] = float(value)
            if metric_name == 'L2':
                summary.append(f'{key}={float(value):.4f}')
    return results, summary


class CustomEvalHook(BaseEvalHook):
    """Non-distributed FarmSim evaluation hook.

    The stock MMDetection hook calls ``mmdet.apis.single_gpu_test`` and
    assumes each model output is a detection-result list.  FarmSim's model
    returns occupancy confusion matrices in a dictionary, so use the
    repository's custom tester and write those metrics directly to the runner
    log buffer.
    """

    def _do_evaluate(self, runner):
        if not self._should_evaluate(runner):
            return

        from projects.mmdet3d_plugin.bevformer.apis.test import custom_single_gpu_test
        results = custom_single_gpu_test(runner.model, self.dataloader, show=False)
        runner.log_buffer.output['eval_iter_num'] = len(self.dataloader)
        if isinstance(results, dict):
            results, summary = _with_occupancy_summaries(
                results, class_names=getattr(self.dataloader.dataset,
                                              'CLASSES', None))
            results, planning_summary = _with_planning_summaries(results)
            summary.extend(planning_summary)
            if summary:
                runner.logger.info('FarmSim validation metrics: ' + ', '.join(summary))
            for name, value in results.items():
                runner.log_buffer.output[name] = value
            runner.log_buffer.ready = True
            return

        key_score = self.evaluate(runner, results)
        if self.save_best and key_score is not None:
            self._save_ckpt(runner, key_score)


def _calc_dynamic_intervals(start_interval, dynamic_interval_list):
    assert mmcv.is_list_of(dynamic_interval_list, tuple)

    dynamic_milestones = [0]
    dynamic_milestones.extend(
        [dynamic_interval[0] for dynamic_interval in dynamic_interval_list])
    dynamic_intervals = [start_interval]
    dynamic_intervals.extend(
        [dynamic_interval[1] for dynamic_interval in dynamic_interval_list])
    return dynamic_milestones, dynamic_intervals


class CustomDistEvalHook(BaseDistEvalHook):

    def __init__(self, *args, dynamic_intervals=None,  **kwargs):
        super(CustomDistEvalHook, self).__init__(*args, **kwargs)
        self.use_dynamic_intervals = dynamic_intervals is not None
        if self.use_dynamic_intervals:
            self.dynamic_milestones, self.dynamic_intervals = \
                _calc_dynamic_intervals(self.interval, dynamic_intervals)

    def _decide_interval(self, runner):
        if self.use_dynamic_intervals:
            progress = runner.epoch if self.by_epoch else runner.iter
            step = bisect.bisect(self.dynamic_milestones, (progress + 1))
            # Dynamically modify the evaluation interval
            self.interval = self.dynamic_intervals[step - 1]

    def before_train_epoch(self, runner):
        """Evaluate the model only at the start of training by epoch."""
        self._decide_interval(runner)
        super().before_train_epoch(runner)

    def before_train_iter(self, runner):
        self._decide_interval(runner)
        super().before_train_iter(runner)

    def _do_evaluate(self, runner):
        """perform evaluation and save ckpt."""
        # Synchronization of BatchNorm's buffer (running_mean
        # and running_var) is not supported in the DDP of pytorch,
        # which may cause the inconsistent performance of models in
        # different ranks, so we broadcast BatchNorm's buffers
        # of rank 0 to other ranks to avoid this.
        if self.broadcast_bn_buffer:
            model = runner.model
            for name, module in model.named_modules():
                if isinstance(module,
                              _BatchNorm) and module.track_running_stats:
                    dist.broadcast(module.running_var, 0)
                    dist.broadcast(module.running_mean, 0)

        if not self._should_evaluate(runner):
            return

        tmpdir = self.tmpdir
        if tmpdir is None:
            tmpdir = osp.join(runner.work_dir, '.eval_hook')

        from projects.mmdet3d_plugin.bevformer.apis.test import custom_multi_gpu_test # to solve circlur  import

        results = custom_multi_gpu_test(
            runner.model,
            self.dataloader,
            tmpdir=tmpdir,
            gpu_collect=self.gpu_collect)
        if runner.rank == 0:
            print('\n')
            runner.log_buffer.output['eval_iter_num'] = len(self.dataloader)

            key_score = self.evaluate(runner, results)

            if self.save_best:
                self._save_ckpt(runner, key_score)

    def evaluate(self, runner, results):
        """Log FarmSim's already-aggregated dictionary metrics directly."""
        if isinstance(results, dict):
            results, summary = _with_occupancy_summaries(
                results, class_names=getattr(self.dataloader.dataset,
                                              'CLASSES', None))
            results, planning_summary = _with_planning_summaries(results)
            summary.extend(planning_summary)
            if summary:
                runner.logger.info('FarmSim validation metrics: ' + ', '.join(summary))
            for name, value in results.items():
                runner.log_buffer.output[name] = value
            runner.log_buffer.ready = True
            return None
        return super().evaluate(runner, results)
  
