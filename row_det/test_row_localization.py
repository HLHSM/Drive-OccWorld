"""Scientific invariants for length-decoupled row localization."""

import unittest

import numpy as np

from row_det.evaluate_row_localization import score_localization


def row(a, b):
    return dict(points=[a, b])


class LocalizationTests(unittest.TestCase):
    bounds = [0, -7.5, 15, 7.5]

    def score(self, gt, pred, **kwargs):
        return score_localization(gt, pred, self.bounds, **kwargs)['0.20']

    def test_short_correct_row_matches_without_claiming_full_length(self):
        m = self.score([row([0, 0], [15, 0])], [row([4, .1], [7, .1])])
        self.assertEqual((m['tp'], m['fp'], m['fn']), (1, 0, 0))
        self.assertAlmostEqual(m['longitudinal_support_coverage'], 3 / 15)

    def test_tiny_or_disjoint_support_does_not_count(self):
        gt = [row([0, 0], [5, 0])]
        for pred in ([row([4, 0], [8, 0])], [row([6, 0], [10, 0])]):
            m = self.score(gt, pred)
            self.assertEqual((m['tp'], m['fp'], m['fn']), (0, 1, 1))

    def test_angle_gate_rejects_a_crossing(self):
        m = self.score([row([0, 0], [15, 0])], [row([4, -.5], [7, .5])])
        self.assertEqual(m['tp'], 0)

    def test_duplicate_and_missed_rows_remain_errors(self):
        m = self.score([row([0, 0], [15, 0]), row([0, 2], [15, 2])],
                       [row([3, .1], [9, .1]), row([3, .12], [9, .12])])
        self.assertEqual((m['tp'], m['fp'], m['fn']), (1, 1, 1))
        self.assertAlmostEqual(m['f1'], .5)
        self.assertAlmostEqual(m['longitudinal_support_coverage'], 6 / 30)

    def test_strict_distance_and_85_percent_over_entire_common_support(self):
        gt = [row([0, 0], [15, 0])]
        self.assertEqual(self.score(gt, [row([0, .2], [15, .2])])['tp'], 0)
        # Same angle (<3deg), but only half the common support is within 0.2m.
        self.assertEqual(self.score(gt, [row([3, .1], [13, .3])])['tp'], 0)

    def test_reverse_endpoint_order_and_rotation(self):
        gt, pred = row([0, 0], [12, 0]), row([9, .1], [3, .1])
        baseline = self.score([gt], [pred])
        angle = np.deg2rad(20)
        matrix = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
        transform = lambda r: dict(points=(np.asarray(r['points']) @ matrix.T).tolist())
        rotated = self.score([transform(gt)], [transform(pred)])
        self.assertEqual(rotated['tp'], 1)
        self.assertAlmostEqual(baseline['longitudinal_support_coverage'], rotated['longitudinal_support_coverage'])

    def test_empty_frames_and_short_gt_still_in_denominator(self):
        pred = [row([0, 0], [5, 0])]
        self.assertEqual(self.score([], pred)['fp'], 1)
        self.assertEqual(self.score(pred, [])['fn'], 1)
        self.assertIsNone(self.score([], [])['f1'])
        m = self.score([row([0, 0], [1, 0])], [row([0, 0], [1, 0])])
        self.assertEqual((m['tp'], m['fp'], m['fn']), (0, 1, 1))

    def test_curved_fit_is_rejected_instead_of_silently_linearized(self):
        with self.assertRaises(ValueError):
            self.score([row([0, 0], [15, 0])], [dict(points=[[1, 0], [4, 1], [8, 0]])])


if __name__ == '__main__':
    unittest.main()
