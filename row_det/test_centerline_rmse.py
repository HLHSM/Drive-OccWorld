"""Known geometric errors, fair support, and missing-output invariants."""

import unittest

import numpy as np

from row_det.evaluate_centerline_rmse import (
    compare_centerlines, pooled_scores, reference_centerline, summarize)


def line(y=0., start=0., stop=10., slope=0.):
    return dict(points=[[start, y + slope * start], [stop, y + slope * stop]])


def row(number, y, field='field', reverse=False):
    points = line(y)['points']
    return dict(id=f'{field}/row_{number}', field_id=field,
                points=points[::-1] if reverse else points)


class CenterlineRmseTests(unittest.TestCase):
    def test_geometric_reference_ignores_order_and_endpoint_direction(self):
        ref = reference_centerline([row(3, 1, reverse=True), row(2, -1)])
        np.testing.assert_allclose(ref['points'], [[0, 0], [10, 0]])
        self.assertEqual(ref['row_ids'], ['field/row_2', 'field/row_3'])

    def test_reference_does_not_bridge_fields_or_invalid_row_gaps(self):
        self.assertFalse(reference_centerline([row(1, -1), row(3, 1)])['points'])
        self.assertFalse(reference_centerline([row(1, -1, 'a'), row(2, 1, 'b')])['points'])
        self.assertFalse(reference_centerline([row(1, 1), row(2, 3)])['points'])

    def test_constant_offset_has_known_rmse_no_threshold_gate(self):
        score = compare_centerlines(line(), {'pred': line(2.5)})
        self.assertEqual(score['status'], 'evaluated')
        self.assertAlmostEqual(score['scores']['pred']['rmse_m'], 2.5)
        self.assertAlmostEqual(score['scores']['pred']['mae_m'], 2.5)

    def test_heading_error_contributes_at_corresponding_x(self):
        score = compare_centerlines(line(), {'pred': line(slope=.1)}, sample_step=1.)
        expected = .1 * np.arange(11)
        self.assertAlmostEqual(score['scores']['pred']['rmse_m'], np.sqrt(np.mean(expected ** 2)))

    def test_both_sides_use_identical_finite_support(self):
        result = compare_centerlines(line(), {'gt': line(.1, 2, 9), 'pred': line(.2, 4, 8)})
        self.assertEqual(result['support_x_m'], [4, 8])
        self.assertEqual(result['samples'], 81)
        self.assertAlmostEqual(result['scores']['gt']['rmse_m'], .1)
        self.assertAlmostEqual(result['scores']['pred']['rmse_m'], .2)

    def test_missing_or_short_output_is_not_zero_error(self):
        result = compare_centerlines(line(), {'pred': dict(points=[])})
        self.assertEqual(result['status'], 'missing_estimate')
        self.assertFalse(result['scores'])
        short = compare_centerlines(line(), {'pred': line(stop=1.)})
        self.assertEqual(short['status'], 'insufficient_common_support')
        self.assertIsNone(pooled_scores([result, short], 'pred')['rmse_m'])
        available = compare_centerlines(line(), {'pred': line(.4)})
        pooled = pooled_scores([result, short, available], 'pred')
        self.assertEqual(pooled['frames'], 1)
        self.assertEqual(pooled['samples'], available['samples'])
        self.assertAlmostEqual(pooled['rmse_m'], .4)

    def test_global_rmse_pools_squared_errors_before_square_root(self):
        a = compare_centerlines(line(), {'pred': line(1.)})
        b = compare_centerlines(line(), {'pred': line(3.)})
        score = pooled_scores([a, b], 'pred')
        self.assertAlmostEqual(score['rmse_m'], np.sqrt(5.))
        self.assertNotAlmostEqual(score['rmse_m'], 2.)

    def test_empty_geometry_and_non_monotone_estimates(self):
        self.assertEqual(compare_centerlines(dict(points=[]), {'pred': line()})['status'], 'no_reference')
        with self.assertRaises(ValueError):
            compare_centerlines(line(), {'pred': dict(points=[[0, 0], [2, 0], [1, 0]])})


if __name__ == '__main__':
    unittest.main()
