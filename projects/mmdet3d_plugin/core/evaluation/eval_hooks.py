
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


def _with_occupancy_summaries(results):
    """Place scalar occupancy metrics before the raw confusion matrices."""
    if not isinstance(results, dict):
        return results, []

    ordered = {}
    raw_results = {}
    summary = []
    for key, value in results.items():
        if key.startswith('hist_for_iou'):
            hist = np.asarray(value)
            # Distributed collection may leave one outer result-list axis.
            if hist.ndim == 3 and hist.shape[0] == 1:
                hist = hist[0]
            if hist.ndim == 2 and hist.shape[0] == hist.shape[1]:
                diagonal = np.diag(hist).astype(np.float64)
                union = hist.sum(axis=1) + hist.sum(axis=0) - diagonal
                valid = union > 0
                iou = np.divide(
                    diagonal, union, out=np.zeros_like(diagonal), where=valid)
                present_miou = float(iou[valid].mean()) if valid.any() else 0.0
                all_miou = float(iou.mean()) if len(iou) else 0.0
                total = float(hist.sum())
                voxel_acc = float(diagonal.sum() / total) if total else 0.0
                metrics = (
                    (f'{key}_mIoU_present', present_miou),
                    (f'{key}_mIoU_all', all_miou),
                    (f'{key}_voxel_acc', voxel_acc),
                )
                for metric_key, metric_value in metrics:
                    ordered[metric_key] = metric_value
                summary.extend(
                    f'{metric_key}={metric_value:.4f}'
                    for metric_key, metric_value in metrics)
        raw_results[key] = value
    # Keep every scalar summary before any large confusion-matrix payload.
    ordered.update(raw_results)
    return ordered, summary


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
            results, summary = _with_occupancy_summaries(results)
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
            results, summary = _with_occupancy_summaries(results)
            if summary:
                runner.logger.info('FarmSim validation metrics: ' + ', '.join(summary))
            for name, value in results.items():
                runner.log_buffer.output[name] = value
            runner.log_buffer.ready = True
            return None
        return super().evaluate(runner, results)
  
