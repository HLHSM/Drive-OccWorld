"""UE poses and grid-to-camera projection regression tests."""
import unittest
import numpy as np
from projects.mmdet3d_plugin.datasets.farmsim_world_dataset import (
    _camera_matrices, _grid_camera_matrices,
    _ue_world_matrix, FarmSimWorldDataset,
)


def pose(location=(0, 0, 0), rpy=(0, 0, 0)):
    return dict(location_cm=list(location), rotation_deg_rpy=list(rpy))


def sensor(world=None):
    return dict(relative_transform_ue=pose((200, 0, 120)),
                world_transform_ue=world or pose((200, 0, 120)),
                intrinsics=dict(fx=600, fy=600, cx=639.5, cy=359.5))


class CameraGeometryTests(unittest.TestCase):
    def test_unreal_positive_pitch_points_nose_up(self):
        np.testing.assert_allclose(_ue_world_matrix(pose(rpy=(0, 90, 0)))[:3, 0],
                                   [0, 0, 1], atol=1e-7)
        np.testing.assert_allclose(_ue_world_matrix(pose(rpy=(90, 0, 0)))[:3, 1],
                                   [0, 0, -1], atol=1e-7)

    def test_world_identity_matches_legacy_mount(self):
        s = sensor()
        meta = dict(semantic_occupancy=dict(grid_frame_world_ue=pose()))
        actual, k = _grid_camera_matrices(s, meta)
        expected, kk = _camera_matrices(s)
        np.testing.assert_allclose(actual, expected)
        np.testing.assert_allclose(k, kk)

    def test_grid_origin_and_camera_world_pose_override_mount(self):
        # Grid world origin (10,20,5), heading 90deg; camera centre in grid
        # coordinates (2,0,1). Deliberately wrong mount/vehicle data are ignored.
        s = sensor(pose((1000, 2200, 600), (0, 0, 90)))
        s['relative_transform_ue'] = pose((9900, 9900, 9900))
        meta = dict(semantic_occupancy=dict(
            grid_frame_world_ue=pose((1000, 2000, 500), (0, 0, 90))),
            vehicle_pose_ue=pose((99999, 99999, 99999)))
        ext, _ = _grid_camera_matrices(s, meta)
        np.testing.assert_allclose(ext @ [7, 2, 0, 1], [2, 1, 5, 1], atol=1e-5)

    def test_missing_world_pose_is_an_error_not_silent_fallback(self):
        s = sensor()
        del s['world_transform_ue']
        with self.assertRaises(KeyError):
            _grid_camera_matrices(s, dict(semantic_occupancy=dict(grid_frame_world_ue=pose())))

    def test_geometry_does_not_read_crop_annotation(self):
        s = sensor(pose((220, -15, 150), (-8, 7, -4)))
        meta = dict(semantic_occupancy=dict(grid_frame_world_ue=pose()))
        before = _grid_camera_matrices(s, meta)
        meta['evaluation_ground_truth'] = {'deliberately': 'unusable'}
        after = _grid_camera_matrices(s, meta)
        for a, b in zip(before, after):
            np.testing.assert_array_equal(a, b)

    def test_frame_motion_uses_the_same_grid_as_projection(self):
        dataset = FarmSimWorldDataset.__new__(FarmSimWorldDataset)
        dataset.camera_names = ('front_rgb',)
        camera = dict(sensor(pose((1000, 2200, 600), (0, 0, 90))),
                      name='front_rgb', type='rgb', enabled=True)
        metadata = dict(sensors=[camera],
            semantic_occupancy=dict(grid_frame_world_ue=pose((1000, 2000, 500), (0, 0, 90))),
            vehicle_center_pose_metric=dict(position_m=[99, 99, 99], quaternion_wxyz=[1, 0, 0, 0]))
        dataset._meta = lambda sequence, frame: metadata
        dataset.camera_geometry = 'grid_world'
        meta, (position, yaw) = dataset._frame_meta(dict(scenario_id='test'), '0', None)
        np.testing.assert_allclose(position, [10, 20, 5])
        self.assertEqual(yaw, 90)
        dataset.camera_geometry = 'legacy'
        _, (position, yaw) = dataset._frame_meta(dict(scenario_id='test'), '0', None)
        np.testing.assert_allclose(position, [99, 99, 99])
        self.assertEqual(yaw, 0)

    def test_invalid_mode_is_rejected_before_dataset_io(self):
        with self.assertRaises(ValueError):
            FarmSimWorldDataset('nonexistent.json', camera_geometry='grid_wrold')


if __name__ == '__main__':
    unittest.main()
