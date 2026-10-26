"""Regressions for MMCV scatter under old and new PyTorch stream APIs.

Set FARMSIM_TEST_CUDA_SCATTER=1 to also exercise real DataContainer transfers.
"""

import importlib
import os
import unittest
from unittest import mock

import torch
import sitecustomize as compat


class StreamCompatibilityTests(unittest.TestCase):
    def check_api(self, uses_device, cases):
        with mock.patch.object(compat, '_torch_stream_uses_device', uses_device), \
                mock.patch.object(compat, '_torch_get_stream', side_effect=lambda x: x), \
                mock.patch.object(torch.cuda, 'current_device', return_value=1):
            for value, expected in cases:
                with self.subTest(value=value, uses_device=uses_device):
                    self.assertEqual(compat._dow2_get_stream(value), expected)

    def test_modern_helper_receives_device_objects(self):
        self.check_api(True, [
            (0, torch.device('cuda:0')), (1, torch.device('cuda:1')),
            (-1, torch.device('cpu')), (torch.device('cpu'), torch.device('cpu')),
            (torch.device('cuda'), torch.device('cuda:1')),
            (torch.device('cuda:0'), torch.device('cuda:0'))])

    def test_legacy_helper_receives_integer_indices(self):
        self.check_api(False, [
            (0, 0), (1, 1), (-1, -1), (torch.device('cpu'), -1),
            (torch.device('cuda'), 1), (torch.device('cuda:0'), 0)])

    def test_reload_preserves_original_helper(self):
        from torch.nn.parallel._functions import _get_stream
        importlib.reload(compat)
        self.assertIs(compat._torch_get_stream, _get_stream)

    @unittest.skipUnless(os.environ.get('FARMSIM_TEST_CUDA_SCATTER') == '1',
                         'opt-in real CUDA transfer test')
    def test_real_mmcv_datacontainer_scatter(self):
        from mmcv.parallel import DataContainer, scatter
        self.assertTrue(torch.cuda.is_available())
        for index in range(torch.cuda.device_count()):
            with self.subTest(gpu=index):
                source = torch.arange(24, dtype=torch.float32).reshape(2, 3, 4)
                inputs = dict(img=DataContainer([source], stack=True),
                              img_metas=DataContainer([[dict(sample='test')]], cpu_only=True))
                output = scatter((inputs,), [index])[0][0]
                self.assertEqual(output['img'].device, torch.device('cuda', index))
                torch.testing.assert_close(output['img'].cpu(), source)
                self.assertEqual(output['img_metas'], [dict(sample='test')])
                self.assertIsNone(compat._dow2_get_stream(-1))
                torch.cuda.synchronize(index)


if __name__ == '__main__':
    unittest.main()
