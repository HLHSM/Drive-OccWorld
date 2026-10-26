"""CPU regressions for SparseOcc's FarmSim geometry and supervision.

Extract the actual pure tensor functions so these tests do not require the
external repository's compiled CUDA extension or OpenMMLab registries.
"""

import ast
from pathlib import Path
import runpy
from types import SimpleNamespace
import unittest

import torch


ROOT = Path(__file__).resolve().parents[2]
SPARSE = ROOT/'experiment/official/SparseOcc'


def load_function(path, name, **dependencies):
    tree = ast.parse(path.read_text())
    node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == name)
    namespace = dict(torch=torch, **dependencies)
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(path), 'exec'), namespace)
    return namespace[name]


class SparseOccFarmSimGeometryTests(unittest.TestCase):
    def test_integer_indices_preserve_metric_coordinates_and_box_sizes(self):
        path = SPARSE/'models/sparse_voxel_decoder.py'
        index2point = load_function(path, 'index2point')
        point2bbox = load_function(path, 'point2bbox')
        coords = torch.tensor([[[0, 0, 0], [40, 48, 16], [72, 72, 24]]])
        points = index2point(coords, [0, -7.5, -2, 15, 7.5, 3], [.2, .2, .2])
        torch.testing.assert_close(points, coords.float()*.2 + torch.tensor([0, -7.5, -2]))
        for interval in (8, 4, 2, 1):
            boxes = point2bbox(points, [.2*interval]*3)
            self.assertTrue(torch.isfinite(boxes[..., 3:].log()).all())
            torch.testing.assert_close(boxes[..., 3:], torch.full_like(points, .2*interval))
        # Also guard direct callers supplying integer positions to point2bbox.
        boxes = point2bbox(coords, [.4, .4, .4])
        torch.testing.assert_close(boxes[..., 3:], torch.full_like(points, .4))

    def test_sampling_uses_the_selected_camera_for_one_three_and_six_views(self):
        def backend(features, locations, weights):
            cameras = (locations[..., 2]*(features[0].shape[1]-1)).round().long()
            return cameras.float()[:, :, None, :]  # BTG,Q,C,P

        sampling = load_function(SPARSE/'models/sparsebev_sampling.py', 'sampling_4d',
                                 DUMP=SimpleNamespace(enabled=False), msmv_sampling=backend)
        for count in (1, 3, 6):
            with self.subTest(cameras=count):
                # Each point is visible in exactly its corresponding camera.
                points = torch.zeros(1, count, 1, 1, 1, 3)
                points[0, :, 0, 0, 0, 0] = torch.arange(count)
                matrices = torch.eye(4).repeat(1, count, 1, 1)
                for i in range(count):
                    matrices[0, i, 0] = torch.tensor([1, 0, 0, .5-i])
                    matrices[0, i, 1] = torch.tensor([0, 0, 0, .5])
                    matrices[0, i, 2] = torch.tensor([0, 0, 0, 1])
                output = sampling(points, [torch.zeros(1, count, 2, 2, 1)],
                                  torch.ones(1, count, 1, 1, 1, 1), matrices, 1, 1)
                torch.testing.assert_close(output.flatten(), torch.arange(count).float())

    def test_coarse_supervision_preserves_integer_blocks_and_gradients(self):
        sparse2dense = load_function(SPARSE/'models/utils.py', 'sparse2dense')
        loss_input = load_function(SPARSE/'models/loss_utils.py',
                                   'get_voxel_decoder_loss_input', sparse2dense=sparse2dense)
        shape = (75, 75, 25)
        target = torch.arange(75*75*25).reshape(1, *shape)
        for scale in (4, 2, 1):
            with self.subTest(scale=scale):
                # Include both the lower corner and the clipped upper block.
                loc = torch.tensor([[[0, 0, 0],
                                     [(n-1)//scale for n in shape]]])
                logits = torch.tensor([[[1., 2.], [3., 4.]]], requires_grad=True)
                pred, gt, mask = loss_input(target, loc, logits, scale, 2, shape)
                axes = torch.meshgrid(*(torch.arange(n)//scale for n in shape), indexing='ij')
                ids = torch.stack(axes, dim=-1)
                expected = ((ids == loc[0, 0]).all(-1) | (ids == loc[0, 1]).all(-1))[None]
                self.assertTrue(torch.equal(mask, expected))
                self.assertTrue(torch.equal(gt, target[expected]))
                expected_pred = torch.where(
                    (ids[expected[0]] == loc[0, 0]).all(-1)[:, None],
                    logits[0, 0], logits[0, 1])
                torch.testing.assert_close(pred, expected_pred)
                pred.sum().backward()
                self.assertTrue(torch.isfinite(logits.grad).all())
                self.assertTrue((logits.grad > 0).all())

    def test_farmsim_schedule_tracks_actual_runner_length(self):
        cfg = runpy.run_path(str(SPARSE/'configs/farmsim_front3_current.py'))
        schedule = cfg['lr_config']
        self.assertEqual(schedule['policy'], 'CosineAnnealing')
        self.assertFalse(schedule['by_epoch'])
        self.assertNotIn('step', schedule)
        self.assertEqual(cfg['total_epochs'], 15)

    def test_cuda_sampling_boundary_promotes_half_inputs(self):
        class Backend:
            @staticmethod
            def apply(*args):
                for arg in args:
                    self.assertEqual(arg.dtype, torch.float32)
                    self.assertTrue(arg.is_contiguous())
                return sum(arg.sum() for arg in args)

        sampling = load_function(SPARSE/'models/csrc/wrapper.py', 'msmv_sampling',
                                 MSMVSamplingC2345=Backend, MSMVSamplingC23456=Backend)
        for levels in (4, 5):
            tensors = [torch.ones(2, 3, dtype=torch.float16, requires_grad=True)
                       for _ in range(levels+2)]
            output = sampling(tensors[:levels], tensors[-2], tensors[-1])
            output.backward()
            self.assertTrue(all(t.grad is not None and torch.isfinite(t.grad).all()
                                for t in tensors))

    def test_online_eval_restores_precision_state_even_on_error(self):
        tree = ast.parse((SPARSE/'models/sparseocc.py').read_text())
        cls = next(node for node in tree.body if isinstance(node, ast.ClassDef)
                   and node.name == 'SparseOcc')
        method = next(node for node in cls.body if isinstance(node, ast.FunctionDef)
                      and node.name == 'simple_test_online')
        ns = {}
        exec(compile(ast.Module(body=[method], type_ignores=[]), '<online-test>', 'exec'), ns)
        for enabled in (False, True):
            for fail in (False, True):
                model = SimpleNamespace(fp16_enabled=enabled)

                def inference(*args):
                    self.assertFalse(model.fp16_enabled)
                    if fail:
                        raise RuntimeError('test inference failure')
                    return 'ok'

                model._simple_test_online = inference
                if fail:
                    with self.assertRaises(RuntimeError):
                        ns['simple_test_online'](model, [])
                else:
                    self.assertEqual(ns['simple_test_online'](model, []), 'ok')
                self.assertEqual(model.fp16_enabled, enabled)


if __name__ == '__main__':
    unittest.main()
