"""Metric invariants, including finite endpoints and cardinality matching."""

import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from row_det.geometric_evaluation import (
    clip_segment, counts_metrics, evaluate_rows, geometric_rows,
    maximum_matching, point_to_finite_segments, sample_segments, world_xy_to_grid,
)


def row(a, b):
    return dict(points=[a, b])


class GeometryMetricTests(unittest.TestCase):
    def test_world_transform_uses_grid_not_vehicle(self):
        grid = dict(location_cm=[1000, 2000, 500], rotation_deg_rpy=[0, 0, 90])
        np.testing.assert_allclose(world_xy_to_grid([[10, 21], [9, 22]], grid),
                                   [[1, 0], [2, 1]], atol=1e-12)
        with self.assertRaises(ValueError):
            world_xy_to_grid([[10, 21]], dict(grid, rotation_deg_rpy=[1, 0, 90]))

    def test_clipping_is_finite_and_includes_boundary(self):
        np.testing.assert_allclose(clip_segment([[-2, 0], [20, 0]], [0, -2, 10, 2]), [[0, 0], [10, 0]])
        self.assertIsNone(clip_segment([[-2, 3], [20, 3]], [0, -2, 10, 2]))
        self.assertIsNone(clip_segment([[-1, 0], [0, 0]], [0, -2, 10, 2]))
        self.assertIsNone(clip_segment([[-2, 0], [-1, 0]], [0, -2, 10, 2]))

    def test_full_length_sampling_rejects_short_overlap(self):
        reference = [row([0, 0], [10, 0])]
        estimates = [row([4, 0], [6, 0])]
        result = evaluate_rows(reference, estimates, [0, -2, 10, 2])
        for metrics in result.values():
            self.assertEqual((metrics['tp'], metrics['fp'], metrics['fn']), (0, 1, 1))
        samples = sample_segments([[[0, 0], [10, 0]]], .05)
        np.testing.assert_allclose(samples[[0, -1]], [[0, 0], [10, 0]])
        self.assertLessEqual(np.linalg.norm(np.diff(samples, axis=0), axis=1).max(), .050000001)

    def test_point_distance_clamps_to_segment_endpoints(self):
        distances = point_to_finite_segments([[12, 0], [-1, 1], [5, .2]], [[[0, 0], [10, 0]]])
        np.testing.assert_allclose(distances, [2, np.sqrt(2), .2])

    def test_thresholds_and_duplicates(self):
        reference = [row([0, 0], [10, 0])]
        estimates = [row([0, .15], [10, .15]), row([0, .15], [10, .15])]
        metrics = evaluate_rows(reference, estimates, [0, -2, 10, 2])
        self.assertEqual(metrics['0.10']['tp'], 0)
        self.assertEqual((metrics['0.20']['tp'], metrics['0.20']['fp']), (1, 1))
        self.assertAlmostEqual(metrics['0.20']['f1'], 2 / 3)

    def test_at_least_85_percent_and_strict_distance(self):
        # Exactly 20 reference samples; an estimate ending at x=16 covers 17.
        reference = [row([0, 0], [19, 0])]
        estimate = [row([0, 0], [16, 0])]
        metric = evaluate_rows(reference, estimate, [0, -2, 19, 2], sample_step=1)['0.10']
        self.assertEqual(metric['tp'], 1)
        self.assertEqual(metric['matches'][0]['coverage'], .85)
        shorter = evaluate_rows(reference, [row([0, 0], [15, 0])], [0, -2, 19, 2], sample_step=1)['0.10']
        self.assertEqual(shorter['tp'], 0)
        exact = evaluate_rows([row([0, 0], [10, 0])], [row([0, .1], [10, .1])], [0, -2, 10, 2])
        self.assertEqual(exact['0.10']['tp'], 0)

    def test_augmenting_match_beats_greedy(self):
        # Ref 0 may use either estimate; ref 1 can use only estimate 0.
        pairs = maximum_matching([[True, True], [True, False]], [[0, 1], [0, 0]])
        self.assertEqual(pairs, [(0, 1), (1, 0)])

    def test_empty_references_and_predictions_are_counted(self):
        self.assertEqual(evaluate_rows([], [row([0, 0], [5, 0])], [0, -1, 10, 1])['0.10']['fp'], 1)
        self.assertEqual(evaluate_rows([row([0, 0], [5, 0])], [], [0, -1, 10, 1])['0.10']['fn'], 1)
        self.assertIsNone(counts_metrics(0, 0, 0)['f1'])

    def test_ignore_invalid_fit_and_transform_then_crop(self):
        with tempfile.TemporaryDirectory() as directory:
            rows = [dict(row_id='good', fit_valid=True, start_world_xy_m=[10, 18], end_world_xy_m=[10, 32]),
                    dict(row_id='invalid', fit_valid=False),
                    dict(row_id='outside', fit_valid=True, start_world_xy_m=[20, 20], end_world_xy_m=[20, 25])]
            layout = dict(complete=True, coordinate_frame='world_ue_xy_m', rows=rows)
            (Path(directory) / 'rows.json').write_text(json.dumps(layout))
            annotation = dict(crop_rows_complete=True, crop_rows_file='rows.json',
                grid_frame_world_ue=dict(location_cm=[1000, 2000, 500], rotation_deg_rpy=[0, 0, 90]))
            metadata = dict(evaluation_ground_truth=annotation,
                            vehicle_pose_ue=dict(location_cm=[0, 0, 0], rotation_deg_rpy=[0, 0, 0]))
            result, diagnostics = geometric_rows(metadata, directory, [0, -2, -2, 10, 2, 3])
            self.assertEqual(len(result), 1)
            np.testing.assert_allclose(result[0]['points'], [[0, 0], [10, 0]], atol=1e-12)
            self.assertEqual(diagnostics['fit_invalid_rows'], 1)


if __name__ == '__main__':
    unittest.main()
