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
RGB_EXTENSIONS = ('.jpg', '.jpeg', '.png')

# Training/evaluation taxonomy after aggregating sparse FarmSim v9 classes.
# The source occupancy files still use their original IDs; ``_load_occupancy``
# applies ``FARMSIM_LABEL_REMAP`` before a sample reaches the model.
FARMSIM_CLASSES = (
    'free', 'crop', 'soil_ground', 'drivable', 'other_vegetation',
    'other_obstacle',
)
FARMSIM_PALETTE = [
    (0, 0, 0), (91, 181, 75), (120, 72, 30), (90, 90, 90),
    (55, 150, 80), (160, 80, 190),
]

# Original IDs: 0 free, 1 crop, 2 soil, 3 drivable, 4 building, 5 fence,
# 6 other vegetation, 7 vehicle, 8 person/animal, 9 other obstacle,
# 10 tree trunk, 11 tree foliage.  Unmapped labels (including 8 and 255)
# are ignored by all occupancy losses and metrics.
FARMSIM_LABEL_REMAP = np.full(256, 255, dtype=np.uint8)
FARMSIM_LABEL_REMAP[[0, 1, 2, 3]] = [0, 1, 2, 3]
FARMSIM_LABEL_REMAP[[6, 11]] = 4
FARMSIM_LABEL_REMAP[[4, 5, 7, 9, 10]] = 5


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

    CLASSES = FARMSIM_CLASSES
    PALETTE = FARMSIM_PALETTE

    def __init__(self, ann_file, data_root=None, queue_length=2, camera_mode='surround',
                 image_size=(640, 360), front_only=False, test_mode=False,
                 future_pred_frame_num=0, future_traj_frame_num=0,
                 predict_trajectory=False, return_ground_height=False,
                 max_samples=None, pipeline=None,
                 **kwargs):
        del pipeline, kwargs
        if camera_mode not in ('surround', 'front'):
            raise ValueError("camera_mode must be 'surround' or 'front'")
        self.queue_length = int(queue_length)
        self.camera_mode = camera_mode
        self.front_only = bool(front_only)
        self.test_mode = test_mode
        self.image_size = tuple(image_size)  # width, height
        self.future_pred_frame_num = int(future_pred_frame_num)
        self.future_traj_frame_num = int(future_traj_frame_num) if predict_trajectory else 0
        self.predict_trajectory = bool(predict_trajectory)
        self.return_ground_height = bool(return_ground_height)
        self.max_samples = None if max_samples is None else int(max_samples)
        if self.max_samples is not None and self.max_samples < 1:
            raise ValueError('max_samples must be positive when specified')
        if self.future_pred_frame_num < 0 or self.future_traj_frame_num < 0:
            raise ValueError('future prediction frame counts must be non-negative')
        self.camera_names = FRONT_RGB_CAMERAS if camera_mode == 'front' else RGB_CAMERAS

        ann_file = Path(ann_file).expanduser()
        with ann_file.open('r', encoding='utf-8') as f:
            manifest = json.load(f)
        # Split manifests contain paths relative to the dataset root.  Keep
        # absolute paths working for old manifests and for users migrating
        # incrementally.  ``data_root`` is intentionally explicit in the
        # training scripts so the same split files work on another machine.
        manifest_root = manifest.get('source_root', '.')
        self.data_root = Path(data_root if data_root is not None else manifest_root).expanduser()
        if not self.data_root.is_absolute():
            self.data_root = (Path.cwd() / self.data_root).resolve()
        self.sequences = manifest['sequences']
        self.samples = []
        for seq_idx, seq in enumerate(self.sequences):
            frame_ids = seq['frame_ids']
            required_future = max(self.future_pred_frame_num,
                                  self.future_traj_frame_num)
            for frame_index in range(
                    self.queue_length,
                    len(frame_ids) - required_future):
                self.samples.append((seq_idx, frame_index))
        if self.max_samples is not None:
            self.samples = self.samples[:self.max_samples]
        if not self.samples:
            raise RuntimeError('No usable FarmSim samples; check split and queue_length.')
        # Compatibility with MMDetection's DistributedGroupSampler.  All
        # FarmSim images share the same fixed input resolution, so one group
        # is both sufficient and semantically correct.
        self.flag = np.zeros(len(self.samples), dtype=np.uint8)

    def __len__(self):
        return len(self.samples)

    def _sequence_path(self, seq):
        path = Path(seq['path']).expanduser()
        return path if path.is_absolute() else self.data_root / path

    def _meta(self, seq, frame_id):
        path = self._sequence_path(seq) / 'meta' / f'{frame_id}.json'
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

    @staticmethod
    def _pose_matrix(pose):
        """World-from-vehicle homogeneous transform for (position, yaw)."""
        pos, yaw = pose
        c, s = np.cos(np.deg2rad(yaw)), np.sin(np.deg2rad(yaw))
        matrix = np.eye(4, dtype=np.float32)
        matrix[:3, :3] = np.array(
            ((c, -s, 0), (s, c, 0), (0, 0, 1)), dtype=np.float32)
        matrix[:3, 3] = np.asarray(pos, dtype=np.float32)
        return matrix

    def _future_transforms(self, seq, current_pose, future_ids):
        """Return future-to-reference and reference-to-future transforms."""
        ref_world = self._pose_matrix(current_pose)
        future_to_ref, ref_to_future = [np.eye(4, dtype=np.float32)], [
            np.eye(4, dtype=np.float32)]
        for frame_id in future_ids:
            future_pose = self._frame_meta(
                seq, frame_id, None)[1]
            future_world = self._pose_matrix(future_pose)
            future2ref = np.linalg.inv(ref_world) @ future_world
            future_to_ref.append(future2ref.astype(np.float32))
            ref_to_future.append(np.linalg.inv(future2ref).astype(np.float32))
        return future_to_ref, ref_to_future

    def _trajectory_targets(self, seq, current_pose, future_ids):
        """Build incremental [dx, dy, dyaw] targets in the current ego frame."""
        current_pos, current_yaw = current_pose
        c, s = np.cos(np.deg2rad(current_yaw)), np.sin(np.deg2rad(current_yaw))
        relative = [np.zeros(3, dtype=np.float32)]
        for frame_id in future_ids:
            future_pos, future_yaw = self._frame_meta(seq, frame_id, None)[1]
            delta = np.asarray(future_pos, dtype=np.float32) - np.asarray(current_pos, dtype=np.float32)
            # UE/FarmSim x is forward and y is right; rotate world delta into
            # the current vehicle frame.
            local_xy = np.array((c * delta[0] + s * delta[1],
                                 -s * delta[0] + c * delta[1]), dtype=np.float32)
            yaw_delta = (future_yaw - current_yaw + 180.0) % 360.0 - 180.0
            relative.append(np.array((local_xy[0], local_xy[1],
                                      np.deg2rad(yaw_delta)), dtype=np.float32))
        return np.diff(np.stack(relative, axis=0), axis=0)

    def _load_images(self, seq, frame_id):
        width, height = self.image_size
        sx, sy = width / 1280.0, height / 720.0
        images = []
        for name in self.camera_names:
            camera_dir = self._sequence_path(seq) / name
            path = next((camera_dir / f'{frame_id}{ext}'
                         for ext in RGB_EXTENSIONS
                         if (camera_dir / f'{frame_id}{ext}').is_file()), None)
            if path is None:
                tried = ', '.join(str(camera_dir / f'{frame_id}{ext}')
                                  for ext in RGB_EXTENSIONS)
                raise FileNotFoundError(f'img file does not exist; tried: {tried}')
            image = mmcv.imread(str(path), flag='color').astype(np.float32)
            image = mmcv.imresize(image, (width, height))
            # Match the project Caffe-style BGR input normalization.
            image = mmcv.imnormalize(image, np.array([103.530, 116.280, 123.675]),
                                     np.ones(3), to_rgb=False)
            images.append(image)
        return images, sx, sy

    def _load_occupancy(self, seq, frame_id):
        seq_path = self._sequence_path(seq)
        raw = np.fromfile(seq_path / 'occupancy' / f'{frame_id}.bin', dtype=np.uint8)
        valid = np.fromfile(seq_path / 'occupancy_valid' / f'{frame_id}.bin', dtype=np.uint8)
        expected = 25 * 100 * 200
        if raw.size != expected or valid.size != expected:
            raise RuntimeError(f'{seq_path}: unexpected occupancy byte count for frame {frame_id}')
        # FarmSim storage [z,y,x] -> model supervision [x,y,z].
        raw = raw.reshape(25, 100, 200).transpose(2, 1, 0)
        valid = valid.reshape(25, 100, 200).transpose(2, 1, 0)
        raw[valid == 0] = 255
        raw = FARMSIM_LABEL_REMAP[raw]
        if self.front_only:
            raw = raw[100:, :, :]
        return torch.from_numpy(raw.astype(np.int64, copy=False))

    def _load_ground_height(self, seq, frame_id):
        """Load explicit UE terrain height as [x, y] meters plus validity."""
        meta = self._meta(seq, frame_id)
        occupancy = meta['semantic_occupancy']
        surface = occupancy['ground_surface']
        seq_path = self._sequence_path(seq)
        index = np.fromfile(seq_path / surface['index_file'], dtype='<u2')
        valid = np.fromfile(seq_path / surface['valid_mask_file'], dtype=np.uint8)
        x_size, y_size, _ = occupancy['dimensions_xyz']
        expected = x_size * y_size
        if index.size != expected or valid.size != expected:
            raise RuntimeError(
                f'{seq_path}: unexpected ground-surface size for frame {frame_id}')
        # Source layout is [y, x], x-fastest; model uses [x, y].
        index = index.reshape(y_size, x_size).transpose(1, 0)
        valid = valid.reshape(y_size, x_size).transpose(1, 0).astype(bool)
        min_z = float(occupancy['min_m_xyz'][2])
        voxel_size = float(occupancy['voxel_size_m'])
        # ``index`` is the first valid layer.  Supervise its voxel center so
        # height targets and decoder z centers use the same convention.
        ground_height = min_z + (index.astype(np.float32) + 0.5) * voxel_size
        if self.front_only:
            ground_height = ground_height[100:, :]
            valid = valid[100:, :]
        return (torch.from_numpy(ground_height),
                torch.from_numpy(valid))

    def __getitem__(self, index):
        seq_idx, frame_index = self.samples[index]
        seq = self.sequences[seq_idx]
        input_frame_ids = seq['frame_ids'][frame_index - self.queue_length:frame_index + 1]
        current_frame_id = input_frame_ids[-1]
        future_occ_ids = seq['frame_ids'][frame_index + 1:
                                          frame_index + 1 + self.future_pred_frame_num]
        future_traj_ids = seq['frame_ids'][frame_index + 1:
                                           frame_index + 1 + self.future_traj_frame_num]
        image_queue, meta_queue, pose_queue, previous_pose = [], [], [], None
        for frame_id in input_frame_ids:
            images, sx, sy = self._load_images(seq, frame_id)
            frame_meta, current_pose = self._frame_meta(seq, frame_id, previous_pose)
            previous_pose = current_pose
            pose_queue.append(current_pose)
            for matrix in frame_meta['lidar2img']:
                matrix[0, :] *= sx
                matrix[1, :] *= sy
            image_queue.append(np.stack(images).transpose(0, 3, 1, 2))
            frame_meta.update(img_shape=[images[0].shape] * len(images),
                              ori_shape=[(720, 1280, 3)] * len(images),
                              pad_shape=[images[0].shape] * len(images))
            meta_queue.append(frame_meta)

        # Relative transforms used by the optional future-occupancy decoder.
        reference_pose = pose_queue[-1]
        reference_world = self._pose_matrix(reference_pose)
        for frame_meta, pose in zip(meta_queue, pose_queue):
            frame_world = self._pose_matrix(pose)
            frame_meta['ref_lidar_to_cur_lidar'] = (
                np.linalg.inv(frame_world) @ reference_world).astype(np.float32)
        future_to_ref, ref_to_future = self._future_transforms(
            seq, reference_pose, future_occ_ids)
        meta_queue[-1]['future2ref_lidar_transform'] = future_to_ref
        meta_queue[-1]['ref2future_lidar_transform'] = ref_to_future

        target_ids = [current_frame_id] + future_occ_ids
        occupancy_ids = [current_frame_id] * self.queue_length + target_ids
        segmentation = torch.stack([
            self._load_occupancy(seq, frame_id) for frame_id in occupancy_ids
        ], dim=0)

        result = dict(
            img=DC(torch.from_numpy(np.stack(image_queue)).float(), stack=True),
            img_metas=DC(meta_queue, cpu_only=True),
            # Stable dataset index used to limit and name saved predictions
            # consistently across single- and multi-GPU evaluation.
            sample_idx=torch.tensor(index, dtype=torch.long),
            # The original detector receives a list after mmcv collation.
            segmentation=segmentation,
        )
        # ``getattr`` also keeps worker processes created from a pre-TGHD
        # dataset instance compatible when source code is updated in place.
        # Fresh instances always define the attribute in ``__init__`` above.
        if getattr(self, 'return_ground_height', False):
            ground_targets = [self._load_ground_height(seq, frame_id)
                              for frame_id in occupancy_ids]
            result['ground_height'] = torch.stack(
                [item[0] for item in ground_targets], dim=0)
            result['ground_valid'] = torch.stack(
                [item[1] for item in ground_targets], dim=0)
        if self.predict_trajectory:
            trajectory = self._trajectory_targets(seq, reference_pose, future_traj_ids)
            steps = self.future_traj_frame_num
            result.update(
                sdc_planning=trajectory,
                # PlanningLoss expects the original nuScenes mask layout
                # [batch, steps, 2]; both channels are valid for FarmSim.
                sdc_planning_mask=np.ones((steps, 2), dtype=np.float32),
                command=np.full(steps, 2, dtype=np.int64),  # forward
                vel_steering=np.zeros((steps, 4), dtype=np.float32),
                # FarmSim has no 3D object boxes; collision supervision is
                # intentionally empty while trajectory regression remains valid.
                gt_future_boxes=DC([None] * steps, cpu_only=True),
                segmentation_bev=np.zeros((steps, 200, 200), dtype=np.float32),
            )
        return result
