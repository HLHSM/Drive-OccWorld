"""Check voxel statistics against the native loader and explicit denominators."""

import csv
import json
from pathlib import Path
import tempfile
import unittest

import sitecustomize  # noqa: F401
import numpy as np

from projects.mmdet3d_plugin.datasets.farmsim_world_dataset import FarmSimWorldDataset
from tools.analyze_simdata_panoptic_voxels import (
    ROI, active_taxonomy, aggregate, remap_counts, scan_sequence, write_table)


class VoxelStatisticsTests(unittest.TestCase):
    def test_histograms_match_native_loader_mask_mapping_and_roi(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            seq = root/'sequence'
            (seq/'occupancy').mkdir(parents=True)
            schema = dict(region_mode='surround', regions=[dict(id='surround',
                min_m_xyz=[-20,-10,-2], max_m_xyz=[20,10,3], dimensions_xyz=[200,100,25])])
            (seq/'occupancy_schema.json').write_text(json.dumps(schema))
            z,y,x = np.indices((25,100,200))
            raw = ((z+y+x) % 11).astype(np.uint8)
            valid = (z >= 10).astype(np.uint8)
            raw[11,20,120] = 255  # Unknown despite a valid terrain mask.
            raw[12,21,121] = 8    # A valid person/animal is class 5.
            np.savez(seq/'occupancy/000000.npz', occupancy=raw, occupancy_valid=valid)
            ids = list(range(11))+[255]
            result = scan_sequence((str(seq),ids))
            self.assertEqual(json.loads(json.dumps(result['source_shape_zyx'])),[25,100,200])
            self.assertEqual(int(result['raw'].sum()),raw.size)
            self.assertEqual(int(result['full'].sum()),raw.size)
            self.assertEqual(int(result['full'][0,-1]),int((valid==0).sum())+1)
            self.assertEqual(result['roi_shape_xyz'],[75,75,25])
            manifest = root/'manifest.json'
            row = dict(path='sequence',frame_ids=['000000'])
            manifest.write_text(json.dumps(dict(sequences=[row])))
            dataset = FarmSimWorldDataset(str(manifest),data_root=tmp,queue_length=0,
                front_only=True,occupancy_target_range=ROI.tolist())
            target = dataset._load_occupancy(row,'000000').numpy()
            names,mapping = active_taxonomy()
            actual = remap_counts(result['roi'],ids,mapping,list(range(6))+[255])[0]
            expected = np.bincount(target.ravel(),minlength=256)[list(range(6))+[255]]
            np.testing.assert_array_equal(actual,expected)

    def test_weighted_global_percentage_differs_from_frame_mean(self):
        names,mapping = active_taxonomy()
        counts = np.array([[80,20,0,0,0,0,0],[0,10,0,0,0,0,90]],dtype=np.int64)
        report = aggregate(counts,list(range(6))+[255],names+['ignore'])
        crop = report['classes'][1]
        self.assertAlmostEqual(crop['pct_valid'],30/110*100)
        self.assertAlmostEqual(crop['frame_mean_pct_valid'],60.)
        self.assertAlmostEqual(crop['pct_all'],15.)
        self.assertAlmostEqual(crop['pct_occupied'],100.)
        self.assertIsNone(report['classes'][-1]['pct_valid'])
        self.assertEqual(report['valid_voxels'],110)

    def test_all_ignored_frame_has_no_valid_denominator(self):
        names,mapping = active_taxonomy()
        counts = np.array([[0,0,0,0,0,0,100]],dtype=np.int64)
        ids = list(range(6))+[255]
        report = aggregate(counts,ids,names+['ignore'])
        self.assertTrue(all(c['pct_valid'] is None for c in report['classes']))
        with tempfile.TemporaryDirectory() as tmp:
            output=Path(tmp)/'counts.csv'
            record=dict(sequence_path='sequence',frame_id='0',crop_type='crop',
                growth_stage='early',time_of_day='noon',weather='sunny',split='train',
                used_by_current_schedule=True)
            write_table(output,[record],counts,ids,names+['ignore'])
            with output.open(encoding='utf-8-sig') as handle:
                row=next(csv.DictReader(handle))
            self.assertEqual(row['class_1_crop_pct_valid'],'')
            self.assertEqual(float(row['class_255_ignore_pct_all']),100.)


if __name__=='__main__':
    unittest.main()
