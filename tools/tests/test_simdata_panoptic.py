"""NPZ/BIN supervision parity and separate held-out test configuration."""

import json
import hashlib
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import sitecustomize  # noqa: F401
import numpy as np
from mmcv import Config

from projects.mmdet3d_plugin.datasets.farmsim_world_dataset import FarmSimWorldDataset
from projects.mmdet3d_plugin.datasets.samplers.group_sampler import CropAwareDistributedGroupSampler
from tools.train import apply_farmsim_options, parse_args


ROOT = Path(__file__).resolve().parents[2]
ROI = [0., -7.5, -2., 15., 7.5, 3.]


class PanopticDataTests(unittest.TestCase):
    def make_sequence(self, root):
        seq = root / 'sequence'
        (seq / 'occupancy').mkdir(parents=True)
        (seq / 'occupancy_valid').mkdir()
        (seq / 'occupancy_schema.json').write_text(json.dumps(dict(
            region_mode='surround', regions=[dict(id='surround',
                dimensions_xyz=[200, 100, 25], min_m_xyz=[-20, -10, -2],
                max_m_xyz=[20, 10, 3])])))
        z, y, x = np.indices((25, 100, 200))
        labels = ((x+y+z) % 12).astype(np.uint8)
        valid = np.ones_like(labels)
        labels[2, 14, 104] = 8
        valid[2, 14, 104] = 0
        labels[3, 15, 105] = 8
        labels[4, 16, 106] = 255
        return seq, labels, valid

    def dataset(self, root):
        manifest = root / 'train.json'
        manifest.write_text(json.dumps(dict(sequences=[dict(
            path='sequence', scenario_id='example', frame_ids=['000001'])])))
        return FarmSimWorldDataset(str(manifest), data_root=str(root),
            queue_length=0, camera_mode='front', front_only=True,
            occupancy_target_range=ROI)

    def test_npz_and_bin_targets_match_with_mask_remap_and_roi(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            seq, labels, valid = self.make_sequence(root)
            labels.tofile(seq / 'occupancy/000001.bin')
            valid.tofile(seq / 'occupancy_valid/000001.bin')
            dataset = self.dataset(root)
            binary = dataset._load_occupancy(dataset.sequences[0], '000001').numpy()
            np.savez_compressed(seq / 'occupancy/000001.npz', occupancy=labels,
                                occupancy_valid=valid,
                                occupancy_instances=np.full_like(labels, 77))
            archive = dataset._load_occupancy(dataset.sequences[0], '000001').numpy()
            self.assertEqual(archive.shape, (75, 75, 25))
            np.testing.assert_array_equal(binary, archive)
            self.assertEqual(archive[4, 1, 2], 255)
            self.assertEqual(archive[5, 2, 3], 5)  # Valid person/animal is an obstacle.
            self.assertEqual(archive[6, 3, 4], 255)  # Unknown geometry stays ignored.
            self.assertLessEqual(set(np.unique(archive)), {0, 1, 2, 3, 4, 5, 255})

    def test_npz_rejects_axis_permutation_even_with_same_voxel_count(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            seq, labels, valid = self.make_sequence(root)
            np.savez(seq / 'occupancy/000001.npz', occupancy=labels.transpose(2, 1, 0),
                     occupancy_valid=valid.transpose(2, 1, 0))
            dataset = self.dataset(root)
            with self.assertRaisesRegex(RuntimeError, 'expected uint8'):
                dataset._load_occupancy(dataset.sequences[0], '000001')

    def test_crop_sampler_stats_read_the_same_npz_target(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            seq, labels, valid = self.make_sequence(root)
            np.savez(seq / 'occupancy/000001.npz', occupancy=labels, occupancy_valid=valid)
            dataset = self.dataset(root)
            target = dataset._load_occupancy(dataset.sequences[0], '000001').numpy()
            output = root / 'stats.json'
            subprocess.run([sys.executable, 'tools/build_farmsim_crop_stats.py',
                '--manifest', str(root / 'train.json'), '--data-root', str(root),
                '--output', str(output), '--target-point-cloud-range', *map(str, ROI)],
                cwd=ROOT, check=True, capture_output=True, text=True)
            self.assertEqual(json.loads(output.read_text())['crop_voxel_counts'],
                             [[int(np.count_nonzero(target == 1))]])

    def test_default_queue_enables_cropsample_and_switches_gvad(self):
        # apply_farmsim_options checks index existence; numerical NPZ/index
        # correctness is covered by test_crop_sampler_stats_read_the_same_npz_target.
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            stats = root / 'stats.json'
            stats.touch()
            capture = root / 'commands.jsonl'
            python_stub = root / 'python'
            python_stub.write_text(
                f'#!{sys.executable}\nimport json, os, sys\n'
                'with open(os.environ["CAPTURE_COMMANDS"], "a") as log:\n'
                '    log.write(json.dumps(sys.argv[1:])+"\\n")\n')
            python_stub.chmod(0o755)
            env = dict(os.environ, CUDA_VISIBLE_DEVICES='0', NUM_GPUS='1',
                       CROP_STATS_FILE=str(stats), PYTHON_BIN=str(python_stub),
                       WORK_ROOT=str(root / 'work_dirs'), CAPTURE_COMMANDS=str(capture))
            env.pop('SAMPLE_STRIDE', None)
            subprocess.run(['bash', 'tools/train_simdata_panoptic_front3_s.sh'],
                cwd=ROOT, env=env, text=True, capture_output=True, check=True)
            commands = [json.loads(line) for line in capture.read_text().splitlines()
                        if 'tools/train.py' in json.loads(line)]
            self.assertEqual(len(commands), 2)
            work_dirs = []
            for use_gvad, command in enumerate(commands):
                with self.subTest(gvad=use_gvad):
                    with patch.object(sys, 'argv', command[command.index('tools/train.py'):]):
                        args = parse_args()
                    cfg = Config.fromfile(str(ROOT / args.config))
                    cfg.merge_from_dict(args.cfg_options)
                    with patch.dict(os.environ, CUDA_VISIBLE_DEVICES='0'):
                        apply_farmsim_options(cfg, args)
                    work_dirs.append(args.work_dir)
                    self.assertEqual(args.use_gvad_attention, use_gvad)
                    self.assertEqual(args.disable_temporal_self_attention, 0)
                    self.assertEqual(args.use_nearfar_bev, 0)
                    self.assertNotIn('--freeze-resolution-transfer-base', command)
                    self.assertEqual(Path(args.load_from).name, 'r101_dcn_fcos3d_pretrain.pth')
                    self.assertEqual(cfg.point_cloud_range, ROI)
                    self.assertEqual(cfg.occ_size, [75, 75, 25])
                    self.assertTrue(cfg.model.future_pred_head.use_agri_amoe_decoder)
                    self.assertEqual(cfg.model.future_pred_frame_num, 0)
                    sampler = cfg.data.shuffler_sampler
                    self.assertEqual(sampler['type'], 'CropAwareDistributedGroupSampler')
                    self.assertEqual(sampler['crop_oversample_ratio'], .1)
                    self.assertEqual(sampler['crop_min_voxels'], 500)
                    self.assertEqual(sampler['crop_stats_file'], str(stats))
                    attention = cfg.model.pts_bbox_head.transformer.encoder.transformerlayers.attn_cfgs
                    expected = ('GeometryVisibleAnchorDeformableAttention' if use_gvad
                                else 'TemporalSelfAttention')
                    self.assertEqual(attention[0]['type'], expected)
                    self.assertEqual(attention[1]['type'], 'SpatialCrossAttention')
                    if use_gvad:
                        self.assertTrue(attention[0]['use_visibility'])
                        self.assertTrue(attention[0]['use_local_deformable'])
                    for split in ('train', 'val', 'test'):
                        data = cfg.data[split]
                        self.assertEqual(data.camera_mode, 'front')
                        self.assertEqual(data.camera_geometry, 'grid_world')
                        self.assertEqual(data.queue_length, 0)
                        self.assertEqual(data.get('sample_stride', 1),
                                         1 if split == 'test' else 2)
                        self.assertEqual(data.occupancy_target_range, ROI)
                        self.assertEqual(data.ann_file, f'data/simdata_panoptic/splits/{split}.json')
            self.assertNotEqual(*work_dirs)

    def test_stride_keeps_sequence_boundaries_and_original_crop_indices(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            sequences = [dict(path='first', frame_ids=list(map(str, range(5)))),
                         dict(path='second', frame_ids=list(map(str, range(4))))]
            manifest = root / 'train.json'
            manifest.write_text(json.dumps(dict(sequences=sequences)))
            full = FarmSimWorldDataset(str(manifest), data_root=tmp, queue_length=0)
            sampled = FarmSimWorldDataset(str(manifest), data_root=tmp,
                                          queue_length=0, sample_stride=2)
            self.assertEqual(len(full), 9)
            self.assertEqual(sampled.samples, [(0, 0), (0, 2), (0, 4), (1, 0), (1, 2)])
            stats = root / 'stats.json'
            stats.write_text(json.dumps(dict(version=1,
                manifest_sha256=hashlib.sha256(manifest.read_bytes()).hexdigest(),
                data_root=str(root), sequence_paths=['first', 'second'],
                crop_voxel_counts=[[10, 2000, 3000, 0, 4000], [9, 8000, 999, 555]])))
            sampler = CropAwareDistributedGroupSampler(sampled, num_replicas=1,
                rank=0, crop_stats_file=str(stats), crop_min_voxels=500)
            self.assertEqual(sampler.crop_counts.tolist(), [10, 3000, 4000, 9, 999])
            self.assertEqual(sampler.positive_indices.tolist(), [1, 2, 4])
            self.assertEqual(len(list(sampler)), len(sampled))
            for stride in (0, -1, 1.5):
                with self.subTest(stride=stride), self.assertRaisesRegex(
                        ValueError, 'sample_stride must be a positive integer'):
                    FarmSimWorldDataset(str(manifest), sample_stride=stride)

    def test_stride_preserves_history_and_future_spacing(self):
        with tempfile.TemporaryDirectory() as tmp:
            manifest = Path(tmp) / 'train.json'
            manifest.write_text(json.dumps(dict(sequences=[dict(
                path='sequence', frame_ids=list(map(str, range(9))))])))
            dataset = FarmSimWorldDataset(str(manifest), queue_length=2,
                future_pred_frame_num=2, sample_stride=2)
            self.assertEqual(dataset.samples, [(0, 2), (0, 4), (0, 6)])
            self.assertEqual(dataset.sequences[0]['frame_ids'], list(map(str, range(9))))


if __name__ == '__main__':
    unittest.main()
