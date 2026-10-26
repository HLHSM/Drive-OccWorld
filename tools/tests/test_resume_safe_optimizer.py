"""Exercise real backward/optimizer/scaler steps on CPU, including resume."""

import logging
import unittest
from types import SimpleNamespace

import torch
from mmcv.runner import GradientCumulativeFp16OptimizerHook

from projects.mmdet3d_plugin.core.hooks.resume_safe_optimizer import (
    ResumeSafeGradientCumulativeFp16OptimizerHook,
    ResumeSafeGradientCumulativeOptimizerHook)


def runner_at(start, stop):
    model = torch.nn.Linear(1, 1, bias=False)
    model.weight.data.fill_(1.)
    return SimpleNamespace(
        iter=start, max_iters=stop, _max_iters=stop, model=model,
        optimizer=torch.optim.SGD(model.parameters(), lr=0.1), meta={},
        logger=logging.getLogger(__name__),
        log_buffer=SimpleNamespace(update=lambda *args: None))


def fp16_hook(hook_type):
    hook = hook_type(cumulative_iters=2, grad_clip=dict(max_norm=35.),
                     loss_scale='dynamic')
    # Real scaling, overflow detection and skipped steps without using a GPU.
    hook.loss_scaler = torch.amp.GradScaler('cpu', init_scale=128.)
    return hook


def step(hook, runner):
    runner.outputs = dict(loss=runner.model(torch.ones(1, 1)).square().mean(),
                          num_samples=1)
    hook.after_train_iter(runner)


class ResumeSafeOptimizerTests(unittest.TestCase):
    def test_reproduces_actual_run_failure_and_fixes_it(self):
        for kind in (GradientCumulativeFp16OptimizerHook,
                     ResumeSafeGradientCumulativeFp16OptimizerHook):
            with self.subTest(hook=kind.__name__):
                hook = fp16_hook(kind)
                runner = runner_at(18893, 40485)
                hook.before_run(runner)
                hook._init(runner)
                # epoch 9 starts at global iteration 8 * 2699 = 21592.
                initial = runner.model.weight.detach().clone()
                for runner.iter in (21592, 21593):
                    step(hook, runner)
                if kind is GradientCumulativeFp16OptimizerHook:
                    self.assertEqual(hook.remainder_iters, 0)
                    self.assertTrue(torch.equal(initial, runner.model.weight))
                    self.assertEqual(hook.loss_scaler.get_scale(), 64.)
                else:
                    self.assertEqual(hook.divisible_iters, 40484)
                    self.assertEqual(hook.remainder_iters, 1)
                    self.assertTrue(torch.isfinite(runner.model.weight).all())
                    self.assertFalse(torch.equal(initial, runner.model.weight))
                    self.assertEqual(hook.loss_scaler.get_scale(), 128.)

    def test_resumes_through_complete_and_partial_final_windows(self):
        for start, stop in ((0, 10), (0, 9), (3, 10), (3, 9), (4, 9), (8, 9)):
            for kind in (ResumeSafeGradientCumulativeOptimizerHook,
                         ResumeSafeGradientCumulativeFp16OptimizerHook):
                with self.subTest(start=start, stop=stop, hook=kind.__name__):
                    runner = runner_at(start, stop)
                    hook = (fp16_hook(kind) if 'Fp16' in kind.__name__ else
                            kind(cumulative_iters=2))
                    if 'Fp16' in kind.__name__:
                        hook.before_run(runner)
                    for runner.iter in range(start, stop):
                        step(hook, runner)
                        self.assertTrue(torch.isfinite(runner.model.weight).all())
                    self.assertLess(runner.model.weight.item(), 1.)

    def test_fresh_run_matches_original_fp16_updates(self):
        trajectories = []
        for kind in (GradientCumulativeFp16OptimizerHook,
                     ResumeSafeGradientCumulativeFp16OptimizerHook):
            runner = runner_at(0, 9)
            hook = fp16_hook(kind)
            hook.before_run(runner)
            values = []
            for runner.iter in range(9):
                step(hook, runner)
                values.append(runner.model.weight.item())
            trajectories.append(values)
        self.assertEqual(*trajectories)

    def test_rejects_zero_scale_checkpoint_instead_of_silent_stall(self):
        runner = runner_at(18893, 40485)
        hook = fp16_hook(ResumeSafeGradientCumulativeFp16OptimizerHook)
        state = hook.loss_scaler.state_dict()
        state['scale'] = 0.
        runner.meta['fp16'] = dict(loss_scaler=state)
        with self.assertRaisesRegex(RuntimeError, 'Invalid FP16 loss scale'):
            hook.before_run(runner)


if __name__ == '__main__':
    unittest.main()
