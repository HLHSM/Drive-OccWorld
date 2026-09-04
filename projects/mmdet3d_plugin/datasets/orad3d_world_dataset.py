"""ORAD-3D monocular occupancy adapter for Drive-OccWorld."""

from pathlib import Path

import mmcv
import numpy as np
import torch
from mmdet.datasets import DATASETS

from .farmsim_world_dataset import FarmSimWorldDataset, _rpy_matrix_deg


ORAD3D_CLASSES = (
    'free', 'road', 'safe-road', 'car', 'people', 'water', 'snow',
    'grass-on-road', 'rock',
)
ORAD3D_PALETTE = [
    (0, 0, 0), (128, 128, 128), (0, 200, 0), (255, 0, 0),
    (255, 128, 0), (0, 128, 255), (230, 230, 255), (80, 180, 80),
    (150, 80, 40),
]

# Model coordinates follow the FarmSim convention (x forward, y right,
# z up), while ORAD-3D stores x right, y forward, z up.  This matrix maps
# model coordinates into ORAD vehicle coordinates before camera projection.
MODEL_TO_ORAD = np.array(
    ((0, 1, 0, 0),
     (1, 0, 0, 0),
     (0, 0, 1, 0),
     (0, 0, 0, 1)), dtype=np.float32)


def _parse_calibration(path):
    values = {}
    with Path(path).open('r', encoding='utf-8') as stream:
        for line in stream:
            name, separator, payload = line.partition(':')
            if separator:
                values[name.strip()] = [float(item) for item in payload.split()]
    if len(values.get('cam_K', ())) != 9:
        raise ValueError(f'{path}: expected 9 cam_K values')
    if len(values.get('cam_RT', ())) != 16:
        raise ValueError(f'{path}: expected 16 cam_RT values')
    intrinsic = np.asarray(values['cam_K'], dtype=np.float32).reshape(3, 3)
    vehicle_to_camera = np.asarray(
        values['cam_RT'], dtype=np.float32).reshape(4, 4)
    cam2img = np.eye(4, dtype=np.float32)
    cam2img[:3, :3] = intrinsic
    return vehicle_to_camera @ MODEL_TO_ORAD, cam2img


@DATASETS.register_module()
class ORAD3DWorldDataset(FarmSimWorldDataset):
    """Sequence-safe, camera-only ORAD-3D semantic occupancy dataset.

    ORAD occupancy files contain sparse known voxels as ``[x, y, z, label]``
    in right/forward/up order.  Unknown voxels remain 255 so they do not become
    false free-space supervision.
    """

    CLASSES = ORAD3D_CLASSES
    PALETTE = ORAD3D_PALETTE
    GRID_SIZE = (100, 100, 16)

    def __init__(self, ann_file, data_root=None, queue_length=0,
                 image_size=(640, 360), test_mode=False,
                 future_pred_frame_num=0, max_samples=None, pipeline=None,
                 **kwargs):
        if int(future_pred_frame_num) != 0:
            raise ValueError(
                'ORAD3DWorldDataset currently supports current occupancy only')
        kwargs.pop('camera_mode', None)
        kwargs.pop('front_only', None)
        # ``tools/train.py`` injects these generic FarmSim controls into all
        # dataset configs.  ORAD is explicitly current-frame only, so consume
        # them before passing our fixed zero/False values to the parent.
        kwargs.pop('future_traj_frame_num', None)
        kwargs.pop('predict_trajectory', None)
        super().__init__(
            ann_file=ann_file,
            data_root=data_root,
            queue_length=queue_length,
            camera_mode='front',
            image_size=image_size,
            front_only=False,
            test_mode=test_mode,
            future_pred_frame_num=0,
            future_traj_frame_num=0,
            predict_trajectory=False,
            max_samples=max_samples,
            pipeline=pipeline,
            **kwargs)
        self.camera_names = ('image_data',)
        self._pose_cache = {}

    def _poses(self, seq):
        seq_path = self._sequence_path(seq)
        cache_key = str(seq_path)
        cached = self._pose_cache.get(cache_key)
        if cached is not None:
            return cached
        poses = {}
        path = seq_path / 'poses.txt'
        if path.is_file():
            with path.open('r', encoding='utf-8') as stream:
                for line_number, line in enumerate(stream, 1):
                    fields = [field.strip() for field in line.split(',')]
                    if len(fields) != 7:
                        raise ValueError(
                            f'{path}:{line_number}: expected timestamp plus 6 pose values')
                    timestamp = fields[0]
                    pose = np.asarray(fields[1:], dtype=np.float32)
                    # ORAD right/forward/up -> model forward/right/up.  The
                    # axis swap reverses yaw sign.
                    position = np.array(
                        (pose[1], pose[0], pose[2]), dtype=np.float32)
                    yaw_deg = -float(np.rad2deg(pose[5]))
                    poses[timestamp] = (position, yaw_deg)
        self._pose_cache[cache_key] = poses
        return poses

    def _frame_meta(self, seq, frame_id, previous_pose):
        seq_path = self._sequence_path(seq)
        lidar2cam, cam2img = _parse_calibration(
            seq_path / 'calib' / f'{frame_id}.txt')
        lidar2img = cam2img @ lidar2cam
        current_pose = self._poses(seq).get(
            frame_id, (np.zeros(3, dtype=np.float32), 0.0))
        position, yaw = current_pose
        can_bus = np.zeros(18, dtype=np.float32)
        if previous_pose is not None:
            can_bus[:3] = position - previous_pose[0]
            can_bus[-1] = yaw - previous_pose[1]
        return dict(
            lidar2img=[lidar2img],
            lidar2cam=[lidar2cam],
            lidar2global_rotation=_rpy_matrix_deg((0, 0, yaw)),
            can_bus=can_bus,
            prev_bev_exists=previous_pose is not None,
            scene_token=seq['scenario_id'],
            lidar_token=frame_id,
        ), current_pose

    def _load_images(self, seq, frame_id):
        path = self._sequence_path(seq) / 'image_data' / f'{frame_id}.png'
        if not path.is_file():
            raise FileNotFoundError(f'ORAD image does not exist: {path}')
        image = mmcv.imread(str(path), flag='color').astype(np.float32)
        original_height, original_width = image.shape[:2]
        width, height = self.image_size
        image = mmcv.imresize(image, (width, height))
        image = mmcv.imnormalize(
            image, np.array([103.530, 116.280, 123.675]), np.ones(3),
            to_rgb=False)
        return [image], width / original_width, height / original_height

    def _load_occupancy(self, seq, frame_id):
        path = self._sequence_path(seq) / 'occupancy' / f'{frame_id}.npy'
        sparse = np.load(str(path), allow_pickle=False)
        if sparse.ndim != 2 or sparse.shape[1] != 4:
            raise ValueError(f'{path}: expected sparse occupancy with shape [N, 4]')
        if not np.issubdtype(sparse.dtype, np.integer):
            if not np.equal(sparse, np.rint(sparse)).all():
                raise ValueError(f'{path}: occupancy values must be integers')
            sparse = np.rint(sparse).astype(np.int64)
        else:
            sparse = sparse.astype(np.int64, copy=False)

        coordinates = sparse[:, :3]
        labels = sparse[:, 3]
        lower_ok = (coordinates >= 0).all(axis=1)
        upper_ok = (coordinates < np.asarray(self.GRID_SIZE)).all(axis=1)
        if not (lower_ok & upper_ok).all():
            bad = sparse[~(lower_ok & upper_ok)][0].tolist()
            raise ValueError(f'{path}: out-of-range occupancy row {bad}')
        if ((labels < 0) | (labels >= len(self.CLASSES))).any():
            bad = int(labels[(labels < 0) | (labels >= len(self.CLASSES))][0])
            raise ValueError(f'{path}: unsupported occupancy label {bad}')

        flat = np.ravel_multi_index(coordinates.T, self.GRID_SIZE)
        order = np.argsort(flat, kind='stable')
        sorted_flat, sorted_labels = flat[order], labels[order]
        duplicate = sorted_flat[1:] == sorted_flat[:-1]
        if duplicate.any() and np.any(
                sorted_labels[1:][duplicate] != sorted_labels[:-1][duplicate]):
            raise ValueError(f'{path}: conflicting labels for one voxel')

        # Sparse coordinates are [right, forward, up]; the public model
        # supervision tensor is [forward, right, up].
        dense = np.full(self.GRID_SIZE, 255, dtype=np.uint8)
        dense[coordinates[:, 1], coordinates[:, 0], coordinates[:, 2]] = labels
        return torch.from_numpy(dense.astype(np.int64, copy=False))
