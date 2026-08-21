"""FarmSim/UE5 dataset adapter for occupancy-only Drive-OccWorld training."""

import json
from pathlib import Path

import mmcv
import numpy as np
import torch
from mmcv.parallel import DataContainer as DC
from mmdet.datasets import DATASETS


RGB_CAMERAS = (
    'front_left_rgb', 'front_rgb', 'front_right_rgb',
    'rear_left_rgb', 'rear_rgb', 'rear_right_rgb',
)
FRONT_RGB_CAMERAS = RGB_CAMERAS[:3]


def _rpy_matrix_deg(rpy):
    """UE local-axis rotation (x forward, y right, z up)."""
    roll, pitch, yaw = np.deg2rad(rpy)
    cr, sr = np.cos(roll), np.sin(roll)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cy, sy = np.cos(yaw), np.sin(yaw)
    rx = np.array(((1, 0, 0), (0, cr, -sr), (0, sr, cr)), dtype=np.float32)
    ry = np.array(((cp, 0, sp), (0, 1, 0), (-sp, 0, cp)), dtype=np.float32)
    rz = np.array(((cy, -sy, 0), (sy, cy, 0), (0, 0, 1)), dtype=np.float32)
    return rz @ ry @ rx


def _camera_matrices(sensor):
    """Return optical-camera extrinsic and homogeneous intrinsic matrices.

    UE camera axes are forward/right/up.  BEVFormer uses pinhole optical axes
    right/down/forward, hence the explicit axis conversion below.
    """
    rel = sensor['relative_transform_ue']
    ego_to_cam_ue = np.eye(4, dtype=np.float32)
    ego_to_cam_ue[:3, :3] = _rpy_matrix_deg(rel['rotation_deg_rpy'])
    ego_to_cam_ue[:3, 3] = np.asarray(rel['location_cm'], dtype=np.float32) / 100.0
    cam_ue_to_optical = np.array(
        ((0, 1, 0, 0), (0, 0, -1, 0), (1, 0, 0, 0), (0, 0, 0, 1)),
        dtype=np.float32)
    ego_to_cam_optical = cam_ue_to_optical @ np.linalg.inv(ego_to_cam_ue)
    intr = sensor['intrinsics']
    cam2img = np.eye(4, dtype=np.float32)
    cam2img[0, 0], cam2img[1, 1] = intr['fx'], intr['fy']
    cam2img[0, 2], cam2img[1, 2] = intr['cx'], intr['cy']
    return ego_to_cam_optical, cam2img


@DATASETS.register_module()
class FarmSimWorldDataset(torch.utils.data.Dataset):
    """Sequence-safe UE5 occupancy dataset.

    ``ann_file`` is the JSON written by ``tools/create_farmsim_split.py``.
    No nuScenes files or CAN bus records are required.  Consecutive frames are
    bundled here rather than in the original nuScenes dataset class.
    """

    def __init__(self, ann_file, queue_length=2, camera_mode='surround',
                 image_size=(640, 360), front_only=False, test_mode=False,
                 pipeline=None, **kwargs):
        del pipeline, kwargs
        if camera_mode not in ('surround', 'front'):
            raise ValueError("camera_mode must be 'surround' or 'front'")
        self.queue_length = int(queue_length)
        self.camera_mode = camera_mode
        self.front_only = bool(front_only)
        self.test_mode = test_mode
        self.image_size = tuple(image_size)  # width, height
        self.camera_names = FRONT_RGB_CAMERAS if camera_mode == 'front' else RGB_CAMERAS

        with open(ann_file, 'r', encoding='utf-8') as f:
            manifest = json.load(f)
        self.sequences = manifest['sequences']
        self.samples = []
        for seq_idx, seq in enumerate(self.sequences):
            frame_ids = seq['frame_ids']
            for frame_index in range(self.queue_length, len(frame_ids)):
                self.samples.append((seq_idx, frame_index))
        if not self.samples:
            raise RuntimeError('No usable FarmSim samples; check split and queue_length.')

    def __len__(self):
        return len(self.samples)

    def _meta(self, seq, frame_id):
        path = Path(seq['path']) / 'meta' / f'{frame_id}.json'
        with path.open('r', encoding='utf-8') as f:
            return json.load(f)

    def _frame_meta(self, seq, frame_id, previous_pose):
        meta = self._meta(seq, frame_id)
        sensors = {x['name']: x for x in meta['sensors'] if x['type'] == 'rgb' and x['enabled']}
        missing = [x for x in self.camera_names if x not in sensors]
        if missing:
            raise RuntimeError(f"{seq['path']} frame {frame_id}: missing cameras {missing}")
        lidar2cam, lidar2img = [], []
        for name in self.camera_names:
            ext, intr = _camera_matrices(sensors[name])
            lidar2cam.append(ext)
            lidar2img.append(intr @ ext)

        pos = np.asarray(meta['vehicle_center_pose_metric']['position_m'], dtype=np.float32)
        quat_wxyz = np.asarray(meta['vehicle_center_pose_metric']['quaternion_wxyz'], dtype=np.float32)
        # A level yaw rotation is sufficient for BEV temporal alignment.
        yaw = np.rad2deg(np.arctan2(2 * (quat_wxyz[0] * quat_wxyz[3] + quat_wxyz[1] * quat_wxyz[2]),
                                    1 - 2 * (quat_wxyz[2] ** 2 + quat_wxyz[3] ** 2)))
        can_bus = np.zeros(18, dtype=np.float32)
        if previous_pose is not None:
            can_bus[:3] = pos - previous_pose[0]
            can_bus[-1] = yaw - previous_pose[1]
        return dict(
            lidar2img=lidar2img,
            lidar2cam=lidar2cam,
            lidar2global_rotation=_rpy_matrix_deg((0, 0, yaw)),
            can_bus=can_bus,
            prev_bev_exists=previous_pose is not None,
            scene_token=seq['scenario_id'],
            lidar_token=frame_id,
        ), (pos, yaw)

    def _load_images(self, seq, frame_id):
        width, height = self.image_size
        sx, sy = width / 1280.0, height / 720.0
        images = []
        for name in self.camera_names:
            path = Path(seq['path']) / name / f'{frame_id}.png'
            image = mmcv.imread(str(path), flag='color').astype(np.float32)
            image = mmcv.imresize(image, (width, height))
            # Match the project Caffe-style BGR input normalization.
            image = mmcv.imnormalize(image, np.array([103.530, 116.280, 123.675]),
                                     np.ones(3), to_rgb=False)
            images.append(image)
        return images, sx, sy

    def _load_occupancy(self, seq, frame_id):
        seq_path = Path(seq['path'])
        raw = np.fromfile(seq_path / 'occupancy' / f'{frame_id}.bin', dtype=np.uint8)
        valid = np.fromfile(seq_path / 'occupancy_valid' / f'{frame_id}.bin', dtype=np.uint8)
        expected = 25 * 100 * 200
        if raw.size != expected or valid.size != expected:
            raise RuntimeError(f'{seq_path}: unexpected occupancy byte count for frame {frame_id}')
        # FarmSim storage [z,y,x] -> model supervision [x,y,z].
        raw = raw.reshape(25, 100, 200).transpose(2, 1, 0)
        valid = valid.reshape(25, 100, 200).transpose(2, 1, 0)
        raw[valid == 0] = 255
        if self.front_only:
            raw = raw[100:, :, :]
        return torch.from_numpy(raw.astype(np.int64, copy=False))

    def __getitem__(self, index):
        seq_idx, frame_index = self.samples[index]
        seq = self.sequences[seq_idx]
        frame_ids = seq['frame_ids'][frame_index - self.queue_length:frame_index + 1]
        image_queue, meta_queue, previous_pose = [], [], None
        for frame_id in frame_ids:
            images, sx, sy = self._load_images(seq, frame_id)
            frame_meta, previous_pose = self._frame_meta(seq, frame_id, previous_pose)
            for matrix in frame_meta['lidar2img']:
                matrix[0, :] *= sx
                matrix[1, :] *= sy
            image_queue.append(np.stack(images).transpose(0, 3, 1, 2))
            frame_meta.update(img_shape=[images[0].shape] * len(images),
                              ori_shape=[(720, 1280, 3)] * len(images),
                              pad_shape=[images[0].shape] * len(images))
            meta_queue.append(frame_meta)

        return dict(
            img=DC(torch.from_numpy(np.stack(image_queue)).float(), stack=True),
            img_metas=DC(meta_queue, cpu_only=True),
            # The original detector receives a list after mmcv collation.
            segmentation=self._load_occupancy(seq, frame_ids[-1]).unsqueeze(0).repeat(
                self.queue_length + 1, 1, 1, 1),
        )
