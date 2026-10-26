"""Keep MMCV accumulation boundaries in the runner's absolute iteration space."""

import math

from mmcv.runner import (HOOKS, GradientCumulativeFp16OptimizerHook,
                         GradientCumulativeOptimizerHook)


class _AbsoluteAccumulationBoundary:
    def _init(self, runner):
        super()._init(runner)
        # MMCV 1.x computes this boundary from (max_iters - iter), then
        # compares it with the absolute runner.iter. On resume this can select
        # remainder_iters == 0 for thousands of iterations (loss / 0).
        self.remainder_iters = runner.max_iters % self.cumulative_iters
        self.divisible_iters = runner.max_iters - self.remainder_iters


@HOOKS.register_module()
class ResumeSafeGradientCumulativeOptimizerHook(
        _AbsoluteAccumulationBoundary, GradientCumulativeOptimizerHook):
    pass


@HOOKS.register_module()
class ResumeSafeGradientCumulativeFp16OptimizerHook(
        _AbsoluteAccumulationBoundary, GradientCumulativeFp16OptimizerHook):
    def _check_scale(self):
        scale = self.loss_scaler.get_scale()
        if not math.isfinite(scale) or scale <= 0:
            raise RuntimeError(
                f'Invalid FP16 loss scale {scale}; training cannot update '
                'parameters. Restore a checkpoint from before the gradient '
                'failure, or load its finite weights into a new training run '
                'with a fresh optimizer and loss scaler. Do not resume the '
                'completed epoch counter of a stalled run.')

    def before_run(self, runner):
        super().before_run(runner)
        self._check_scale()

    def after_train_iter(self, runner):
        self._check_scale()
        super().after_train_iter(runner)
        self._check_scale()
