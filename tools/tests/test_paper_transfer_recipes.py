"""CPU checks for the five paper-only transfer recipes; no GPU jobs launched."""

import os
import pickle
import shlex
import subprocess
import tempfile
import unittest
from pathlib import Path

import sitecustomize  # noqa: F401
import numpy as np
import torch
from mmcv import Config

from tools.adapt_farmsim_checkpoint_to_0p1 import resized_tensor
from tools.summarize_orad3d_paper_transfer import summarize


ROOT = Path(__file__).resolve().parents[2]
NATIVE = 'projects/configs/farmsim/farmsim_occ_front3_s_0p1_tsa_amoe.py'
REFINE = 'projects/configs/farmsim/farmsim_occ_front3_s_0p1_headrefine_tsa_amoe.py'
ORAD = 'projects/configs/orad3d/orad3d_occ_mono_tsa_amoe.py'
SOURCE_ROI = [0., -7.5, -2., 15., 7.5, 3.]


class PaperTransferRecipeTests(unittest.TestCase):
    def test_target_architectures_keep_tsa_amoe_only(self):
        for path in (NATIVE, REFINE, ORAD):
            with self.subTest(config=path):
                cfg = Config.fromfile(str(ROOT / path))
                head = cfg.model.future_pred_head
                encoder = cfg.model.pts_bbox_head.transformer.encoder
                self.assertEqual(cfg.model.img_backbone.depth, 101)
                self.assertEqual(encoder.transformerlayers.attn_cfgs[0].type,
                                 'TemporalSelfAttention')
                self.assertFalse(encoder.use_nearfar_bev)
                self.assertTrue(head.use_agri_amoe_decoder)
                self.assertEqual(head.agri_amoe_channels, 96)
                self.assertFalse(head.use_dual_hardness_refinement)
                self.assertEqual(head.history_queue_length, 0)
                for split in ('train', 'val', 'test'):
                    self.assertEqual(cfg.data[split].queue_length, 0)

    def test_resolution_transfer_uses_same_physical_region(self):
        for path, queries in ((NATIVE, 150), (REFINE, 75)):
            cfg = Config.fromfile(str(ROOT / path))
            self.assertEqual(cfg.model.point_cloud_range, SOURCE_ROI)
            self.assertEqual(cfg.occ_size, [150, 150, 25])
            self.assertEqual(cfg.model.bev_h, queries)
            self.assertEqual(cfg.model.bev_w, queries)
            for split in ('train', 'val', 'test'):
                self.assertEqual(cfg.data[split].occupancy_target_range, SOURCE_ROI)
        cfg = Config.fromfile(str(ROOT / REFINE))
        self.assertTrue(cfg.model.future_pred_head.use_occupancy_2x_refiner)

    def test_orad_keeps_official_label_space_and_grid(self):
        cfg = Config.fromfile(str(ROOT / ORAD))
        self.assertEqual(cfg.occ_size, [100, 100, 16])
        self.assertEqual(cfg.model.future_pred_head.num_classes, 9)
        self.assertEqual(cfg.model.pts_bbox_head.transformer.num_cams, 1)
        self.assertEqual(cfg.model.point_cloud_range,
                         [5.7, -25., -3., 55.7, 25., 5.])

    def test_source_queries_resize_to_both_target_lattices(self):
        for side in (100, 150):
            query = torch.ones(75 * 75, 4)
            target = torch.empty(side * side, 4)
            result = resized_tensor('pts_bbox_head.bev_embedding.weight', query, target)
            torch.testing.assert_close(result, torch.ones_like(target))
            position = resized_tensor(
                'pts_bbox_head.positional_encoding.row_embed.weight',
                torch.ones(75, 4), torch.empty(side, 4))
            torch.testing.assert_close(position, torch.ones(side, 4))

    def _dry_run(self, script, **overrides):
        # Stub files make command-construction tests independent of the large
        # private datasets and checkpoints. DRY_RUN must never read their data.
        with tempfile.TemporaryDirectory(prefix='paper_transfer_test_') as tmp:
            root = Path(tmp)
            for split in ('training', 'validation', 'testing'):
                (root / split).mkdir()
            for name in ('train_100.json', 'train.json', 'val.json',
                         'test.json', 'test_farm.json', 'source.pth'):
                (root / name).touch()
            env = os.environ.copy()
            env.update(DRY_RUN='1', RUN_ID='recipe_test', CUDA_VISIBLE_DEVICES='0,1',
                       NUM_GPUS='2', BATCH_SIZE='1', HEADREFINE_BATCH_SIZE='1',
                       TOTAL_BATCH_SIZE='24', ORAD_ROOT=tmp, DATA_ROOT=tmp,
                       SPLIT_DIR=tmp, TRAIN_ANN_FILE=str(root/'train_100.json'),
                       VAL_ANN_FILE=str(root/'val.json'),
                       TEST_ANN_FILE=str(root/'test.json'),
                       FARM_TEST_ANN_FILE=str(root/'test_farm.json'),
                       GENERIC_PRETRAINED=str(root/'source.pth'),
                       FARMSIM_CHECKPOINT=str(root/'source.pth'),
                       IMAGE_PRETRAINED=str(root/'source.pth'),
                       SOURCE_02_CHECKPOINT=str(root/'source.pth'),
                       PREPARE_SPLITS='0', RUN_TEST='1', RUN_SCRATCH_100='1',
                       RUN_FINETUNE_100='1', RUN_SCRATCH='1',
                       RUN_FROZEN_FINETUNE='1', RUN_FROZEN_HEADREFINE='1',
                       EPOCHS='8', SCRATCH_EPOCHS='6', FINETUNE_EPOCHS='2')
            env.update(overrides)
            result = subprocess.run(['bash', str(ROOT / script)], cwd=ROOT,
                                    env=env, capture_output=True, text=True)
            return result

    @staticmethod
    def _commands(output, executable):
        commands = [shlex.split(line[len('[dry-run] '):])
                    for line in output.splitlines() if line.startswith('[dry-run] ')]
        return [cmd for cmd in commands if executable in cmd]

    def test_orad_script_launches_two_trainings_and_four_heldout_tests(self):
        result = self._dry_run('tools/train_orad3d_ablation.sh')
        self.assertEqual(result.returncode, 0, result.stderr)
        train = self._commands(result.stdout, 'tools/train.py')
        tests = self._commands(result.stdout, 'tools/test.py')
        self.assertEqual(len(train), 2)
        self.assertEqual([cmd[cmd.index('--epochs')+1] for cmd in train], ['6', '2'])
        self.assertEqual(len(tests), 4)
        for cmd in tests:
            self.assertIn('work_dirs/orad3d_front3_s_tsa_amoe_',
                          cmd[cmd.index('tools/test.py')+1])
        self.assertNotIn('train_010.json', result.stdout)
        self.assertNotIn('farm_all.json', result.stdout)

    def test_resolution_script_launches_only_three_paper_rows(self):
        result = self._dry_run('tools/train_simdata_occ0p1_resolution_transfer.sh')
        self.assertEqual(result.returncode, 0, result.stderr)
        train = self._commands(result.stdout, 'tools/train.py')
        self.assertEqual(len(train), 3)
        self.assertNotIn('--freeze-resolution-transfer-base', train[0])
        self.assertTrue(all('--freeze-resolution-transfer-base' in cmd for cmd in train[1:]))
        self.assertTrue(all(cmd[cmd.index('--epochs')+1] == '8' for cmd in train))
        self.assertTrue(all(cmd[cmd.index('--batch-size')+1] == '1' for cmd in train))

    def test_invalid_batch_is_rejected_before_launch(self):
        result = self._dry_run('tools/train_simdata_occ0p1_resolution_transfer.sh',
                               BATCH_SIZE='6', TOTAL_BATCH_SIZE='4')
        self.assertNotEqual(result.returncode, 0)
        self.assertEqual(self._commands(result.stdout, 'tools/train.py'), [])

    def test_paper_metrics_use_official_and_heldout_farm_conventions(self):
        with tempfile.TemporaryDirectory(prefix='orad_metrics_test_') as tmp:
            root = Path(tmp)
            hist = np.zeros((9, 9), dtype=np.int64)
            hist[0, 0] = 10
            hist[1, 1] = 5
            for name in ('test_metrics.pkl', 'test_farm_metrics.pkl'):
                with (root / name).open('wb') as handle:
                    pickle.dump({'hist_for_iou_current': hist[None]}, handle)
            result = summarize(root, 120)
            self.assertEqual(result['training_minutes'], 2.)
            values = result['paper_columns_percent']
            self.assertEqual(values['Test (w/o Free)'], 100.)
            self.assertAlmostEqual(values['Test'], 200./9)
            self.assertEqual(values['Test_farm'], 100.)


if __name__ == '__main__':
    unittest.main()
