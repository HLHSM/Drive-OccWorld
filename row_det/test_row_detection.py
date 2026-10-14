"""Behaviour checks for geometry, path suppression, and viewer artifact pairing.

Run: python -m unittest row_det.test_row_detection
"""

import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from row_det.detector import RowConfig, detect_rows, nearest_cells
from row_det.generate_row_predictions import build_artifact, compare_rows, write_json
from tools.visualize_farmsim_predictions import load_sample


PC_RANGE = np.array([0, -7.5, -2, 15, 7.5, 3], dtype=float)


def planted_field(offsets=(-1.2, 1.2), angle=0.0, curved=False, gaps=False):
    labels = np.zeros((75, 75, 25), dtype=np.uint8)
    labels[:, :, 9] = 2  # ground surface top = z=0
    for i in range(5, 70):
        x = (i + .5) * .2
        if gaps and 5 < x < 6.5:
            continue
        for offset in offsets:
            y = np.tan(np.deg2rad(angle)) * x + offset
            if curved:
                y += .008 * (x - 7.5) ** 2
            j = int(np.floor((y + 7.5) / .2))
            if 0 <= j < 75:
                labels[i, j, 10:14] = 1
    return labels


class GeometryTests(unittest.TestCase):
    def test_metric_clearance_matches_brute_force(self):
        mask = np.zeros((7, 11), dtype=bool)
        mask[1, 2] = mask[5, 8] = True
        pitch = np.array([.3, .2])
        distance, nearest = nearest_cells(mask, pitch)
        query = np.indices(mask.shape).transpose(1, 2, 0)
        sites = np.argwhere(mask)
        expected = np.linalg.norm((query[:, :, None] - sites) * pitch, axis=-1).min(axis=-1)
        np.testing.assert_allclose(distance, expected)
        self.assertTrue(mask[tuple(nearest)].all())

    def test_straight_rows_with_missing_plants_and_metric_path(self):
        result = detect_rows(planted_field(gaps=True), PC_RANGE,
                             RowConfig(vehicle_width=.6))
        self.assertEqual(len(result['rows']), 2)
        reference = result['reference_path']
        self.assertEqual(reference['status'], 'clearance_checked')
        points = np.asarray(reference['points'])
        self.assertGreater(np.ptp(points[:, 0]), 10)
        self.assertLess(np.abs(points[:, 1]).max(), .11)
        self.assertTrue(np.allclose(points[:, 2], .05))

    def test_rotated_rows(self):
        result = detect_rows(planted_field(angle=20), PC_RANGE)
        self.assertEqual(len(result['rows']), 2)
        self.assertAlmostEqual(result['diagnostics']['direction_deg'], 20, delta=2)
        self.assertTrue(result['reference_path']['points'])
        self.assertEqual(result['reference_path']['status'], 'geometry_only')

    def test_curved_rows(self):
        result = detect_rows(planted_field(curved=True), PC_RANGE,
                             RowConfig(curve_degree=2, row_band=.5))
        self.assertEqual(len(result['rows']), 2)
        path = np.asarray(result['reference_path']['points'])
        self.assertGreater(len(path), 10)
        expected = .008 * (path[:, 0] - 7.5) ** 2
        self.assertLess(np.abs(path[:, 1] - expected).mean(), .12)

    def test_empty_and_insufficient_crops(self):
        labels = planted_field(offsets=())
        self.assertEqual(detect_rows(labels, PC_RANGE)['rows'], [])
        labels[10, 30, 10] = 1
        self.assertEqual(detect_rows(labels, PC_RANGE)['reference_path']['points'], [])

    def test_vehicle_outside_corridor_has_no_reference(self):
        result = detect_rows(planted_field(offsets=(1, 3)), PC_RANGE)
        self.assertEqual(len(result['rows']), 2)
        self.assertEqual(result['reference_path']['status'], 'no_bracketing_rows')

    def test_wide_vehicle_and_unknown_ground_suppress_reference(self):
        labels = planted_field(offsets=(-.6, .6))
        result = detect_rows(labels, PC_RANGE, RowConfig(vehicle_width=1.2))
        self.assertEqual(len(result['rows']), 2)
        self.assertFalse(result['reference_path']['points'])
        labels[labels == 2] = 0
        self.assertFalse(detect_rows(labels, PC_RANGE)['reference_path']['points'])

    def test_obstacle_truncates_path_without_jumping(self):
        labels = planted_field()
        labels[40:44, 32:43, 10:15] = 5
        path = np.asarray(detect_rows(labels, PC_RANGE,
                                     RowConfig(vehicle_width=.6))['reference_path']['points'])
        self.assertGreater(len(path), 5)
        self.assertLess(path[-1, 0], 8.0)

    def test_underground_ignore_is_not_an_obstacle_but_unknown_air_is(self):
        labels = planted_field()
        labels[:, :, :9] = 255
        self.assertTrue(detect_rows(labels, PC_RANGE)['reference_path']['points'])
        labels[5:15, 32:43, 10:13] = 255
        self.assertFalse(detect_rows(labels, PC_RANGE)['reference_path']['points'])

    def test_same_rows_match_and_distant_rows_do_not(self):
        result = detect_rows(planted_field(), PC_RANGE)
        self.assertEqual(compare_rows(result, result)['matched_rows'], 2)
        shifted = detect_rows(planted_field(offsets=(-3, 3)), PC_RANGE)
        self.assertEqual(compare_rows(result, shifted)['matched_rows'], 0)


class ArtifactTests(unittest.TestCase):
    def test_prediction_independent_of_gt_and_future_viewer_overlay(self):
        with tempfile.TemporaryDirectory() as root:
            prediction = Path(root) / 'prediction'
            prediction.mkdir()
            rows = Path(root) / 'row_prediction'
            rows.mkdir()
            path = prediction / 'sample.npz'
            labels = planted_field()
            np.savez_compressed(path, current_pred=labels,
                                current_gt=np.full_like(labels, 255),
                                future_pred=labels[None], future_gt=labels[None],
                                point_cloud_range=PC_RANGE, num_classes=6)
            artifact = build_artifact(path, RowConfig())
            self.assertEqual(len(artifact['current']['pred']['rows']), 2)
            self.assertFalse(artifact['current']['gt']['rows'])
            write_json(rows / 'sample.json', artifact)
            payload = load_sample(path, 100)
            self.assertEqual(payload['row_overlay_status'], 'Row result loaded')
            self.assertEqual(len(payload['current']['row_prediction']['pred']['rows']), 2)
            self.assertEqual(len(payload['future'][0]['row_prediction']['gt']['rows']), 2)
            # A replaced occupancy result must not silently use old row lines.
            artifact['source']['mtime_ns'] -= 1
            write_json(rows / 'sample.json', artifact)
            payload = load_sample(path, 100)
            self.assertIn('Stale', payload['row_overlay_status'])
            self.assertNotIn('row_prediction', payload['current'])

    def test_missing_and_corrupt_overlay_keep_occupancy_available(self):
        with tempfile.TemporaryDirectory() as root:
            path = Path(root) / 'sample.npz'
            labels = planted_field()
            np.savez(path, current_pred=labels, current_gt=labels, point_cloud_range=PC_RANGE)
            row_dir = Path(root) / 'rows'
            self.assertEqual(load_sample(path, 100, row_dir)['row_overlay_status'],
                             'No row result for this sample')
            row_dir.mkdir()
            (row_dir / 'sample.json').write_text('broken json')
            payload = load_sample(path, 100, row_dir)
            self.assertIn('Cannot load', payload['row_overlay_status'])
            json.dumps(payload, allow_nan=False)


if __name__ == '__main__':
    unittest.main()
