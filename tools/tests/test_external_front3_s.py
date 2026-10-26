"""Geometry and launcher checks for the reduced-ROI external baselines."""

import json
import os
import pickle
import runpy
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

from experiment.common.farmsim_contract import (
    OCC_SIZE, PC_RANGE, _source_grid_and_target_indices, load_common_occupancy)


ROOT = Path(__file__).resolve().parents[2]


class ExternalFront3STests(unittest.TestCase):
    def test_common_labels_follow_native_target_center_rule(self):
        with tempfile.TemporaryDirectory(prefix='front3_s_grid_') as tmp:
            seq = Path(tmp)
            (seq/'occupancy').mkdir()
            (seq/'occupancy_valid').mkdir()
            schema = dict(region_mode='test', regions=[dict(
                id='test', dimensions_xyz=[200,100,25],
                min_m_xyz=[-20,-10,-2], max_m_xyz=[20,10,3])])
            (seq/'occupancy_schema.json').write_text(json.dumps(schema))
            z,y,x = np.indices((25,100,200))
            raw = ((x+y+z) % 4).astype(np.uint8)
            valid = np.ones_like(raw)
            valid[2,14,104] = 0
            raw[3,15,105] = 8  # An unmapped label is ignored, not class 0.
            raw.tofile(seq/'occupancy/000000.bin')
            valid.tofile(seq/'occupancy_valid/000000.bin')
            actual,mask = load_common_occupancy(seq,'000000')
            expected = raw.transpose(2,1,0)[100:175,13:88].copy()
            expected[4,1,2] = 255
            expected[5,2,3] = 255
            self.assertEqual(actual.shape,(75,75,25))
            np.testing.assert_array_equal(actual,expected)
            np.testing.assert_array_equal(mask,expected!=255)
        _source_grid_and_target_indices.cache_clear()

    def test_external_leaf_configs_share_exact_roi(self):
        configs = [
            'experiment/official/COTR/configs/cotr/cotr_farmsim_front3.py',
            'experiment/official/SurroundOcc/projects/configs/surroundocc/surroundocc_farmsim_front3.py',
            'experiment/official/SparseOcc/configs/farmsim_front3_current.py',
        ]
        for path in configs:
            with self.subTest(config=path):
                cfg = runpy.run_path(str(ROOT/path))
                self.assertEqual(tuple(cfg['point_cloud_range']),PC_RANGE)
                grid = cfg.get('occ_size', cfg.get('voxel_config',{}).get('voxel_size'))
                self.assertEqual(tuple(grid),OCC_SIZE)
        cotr = runpy.run_path(str(ROOT/configs[0]))
        self.assertEqual(cotr['model']['img_bev_encoder_neck']['size'],(25,19,19))
        self.assertAlmostEqual(15/cotr['occformer_grid_config']['x'][2],19)

    def test_launcher_preserves_all_methods_and_uses_separate_outputs(self):
        for method in ('driveocc','irwm','cotr','surroundocc','sparseocc','all'):
            with self.subTest(method=method):
                env=os.environ.copy()
                env.update(DRY_RUN='1',CUDA_VISIBLE_DEVICES='1,2,3',NUM_GPUS='3',
                           BATCH_SIZE='4',TOTAL_BATCH_SIZE='24',RUN_EXISTING_IRWM='1',
                           PREPARE_INFOS='1')
                result=subprocess.run(['bash','experiment/train_farmsim_all.sh',method],
                                      cwd=ROOT,env=env,text=True,capture_output=True)
                self.assertEqual(result.returncode,0,result.stderr)
                self.assertIn('front3_s',result.stdout)
                if method in ('driveocc','irwm'):
                    self.assertIn('farmsim_occ_front3_s.py',result.stdout)
                else:
                    self.assertIn('infos_front3_s',result.stdout)
                    self.assertIn('--reuse-if-current',result.stdout)

    def test_info_converter_cache_checks_roi_and_manifest(self):
        with tempfile.TemporaryDirectory(prefix='front3_s_infos_') as tmp:
            root=Path(tmp)
            manifest=root/'train.json'
            manifest.write_text(json.dumps(dict(split='train',sequences=[])))
            output=root/'infos.pkl'
            command=[sys.executable,str(ROOT/'experiment/common/create_farmsim_infos.py'),
                     '--manifest',str(manifest),'--data-root',str(root),
                     '--output',str(output),'--reuse-if-current']
            first=subprocess.run(command,capture_output=True,text=True)
            self.assertEqual(first.returncode,0,first.stderr)
            with output.open('rb') as handle: meta=pickle.load(handle)['metadata']
            self.assertEqual(tuple(meta['occ_size']),OCC_SIZE)
            second=subprocess.run(command,capture_output=True,text=True)
            self.assertEqual(second.returncode,0,second.stderr)
            self.assertIn('Reusing',second.stdout)
            manifest.write_text(json.dumps(dict(split='val',sequences=[])))
            third=subprocess.run(command,capture_output=True,text=True)
            self.assertNotEqual(third.returncode,0)
            self.assertIn('does not match',third.stderr)

    def test_cotr_resume_restores_saved_config_and_ema(self):
        with tempfile.TemporaryDirectory(prefix='cotr_resume_') as tmp:
            work_dir = Path(tmp)
            checkpoint = work_dir/'epoch_2.pth'
            checkpoint.touch()
            (work_dir/'epoch_2_ema.pth').touch()
            (work_dir/'latest.pth').symlink_to(checkpoint.name)
            (work_dir/'cotr_farmsim_front3.py').touch()
            env = os.environ.copy()
            env.update(DRY_RUN='1', PREPARE_INFOS='0', CUDA_VISIBLE_DEVICES='1,2,3',
                       NUM_GPUS='3', BATCH_SIZE='4', TOTAL_BATCH_SIZE='24',
                       COTR_RESUME_FROM='auto', COTR_WORK_DIR=tmp,
                       RUN_EXISTING_IRWM='1', EPOCHS='15')
            # The user's editable `all` queue may temporarily disable COTR.
            command = ['bash', 'experiment/train_farmsim_all.sh', 'cotr']
            result = subprocess.run(command, cwd=ROOT, env=env, text=True,
                                    capture_output=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            output = result.stdout
            self.assertIn(f'--resume-from {checkpoint}', output)
            self.assertIn(f'custom_hooks.0.resume={work_dir}/epoch_2_ema.pth', output)
            self.assertIn(f'{work_dir}/cotr_farmsim_front3.py --launcher', output)
            self.assertIn('runner.max_epochs=15', output)
            (work_dir/'epoch_2_ema.pth').unlink()
            missing = subprocess.run(command, cwd=ROOT, env=env, text=True,
                                     capture_output=True)
            self.assertNotEqual(missing.returncode, 0)
            self.assertIn('EMA checkpoint', missing.stderr)


if __name__=='__main__':
    unittest.main()
