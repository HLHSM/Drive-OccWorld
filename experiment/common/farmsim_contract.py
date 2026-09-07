"""The one FarmSim data contract used by all external occupancy baselines.

The native Drive-OccWorld adapter is deliberately the reference for geometry:
all returned occupancy tensors use ``[x, y, z]`` order in the forward 20 m
crop.  Labels are the common evaluation labels: free=0, crop=1, soil=2,
drivable=3, vegetation=4, obstacle=5 and ignore=255.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Iterable, Tuple

import numpy as np


RGB_CAMERAS = ('front_left_rgb', 'front_rgb', 'front_right_rgb')
OCC_SIZE = (100, 100, 25)
PC_RANGE = (0.0, -10.0, -2.0, 20.0, 10.0, 3.0)
VOXEL_SIZE = (0.2, 0.2, 0.2)
CLASS_NAMES = ('free', 'crop', 'soil_ground', 'drivable',
               'other_vegetation', 'other_obstacle')

_RAW_TO_COMMON = np.full(256, 255, dtype=np.uint8)
_RAW_TO_COMMON[[0, 1, 2, 3]] = [0, 1, 2, 3]
_RAW_TO_COMMON[[6, 11]] = 4
_RAW_TO_COMMON[[4, 5, 7, 9, 10]] = 5


def read_manifest(path: str | Path) -> Dict:
    with Path(path).open('r', encoding='utf-8') as stream:
        return json.load(stream)


def resolve_sequence(data_root: str | Path, sequence: Dict) -> Path:
    path = Path(sequence['path']).expanduser()
    return path if path.is_absolute() else Path(data_root).expanduser() / path


def load_common_occupancy(sequence_path: str | Path, frame_id: str) -> Tuple[np.ndarray, np.ndarray]:
    """Read FarmSim's on-disk [z, y, x] target into the common front grid."""
    sequence_path = Path(sequence_path)
    raw = np.fromfile(sequence_path / 'occupancy' / f'{frame_id}.bin', dtype=np.uint8)
    valid = np.fromfile(sequence_path / 'occupancy_valid' / f'{frame_id}.bin', dtype=np.uint8)
    expected = 25 * 100 * 200
    if raw.size != expected or valid.size != expected:
        raise RuntimeError(f'Unexpected occupancy size in {sequence_path} frame {frame_id}')
    raw = raw.reshape(25, 100, 200).transpose(2, 1, 0)[100:, :, :]
    valid = valid.reshape(25, 100, 200).transpose(2, 1, 0)[100:, :, :].astype(bool)
    labels = _RAW_TO_COMMON[raw]
    labels[~valid] = 255
    return labels, valid


def rpy_matrix_deg(rpy: Iterable[float]) -> np.ndarray:
    roll, pitch, yaw = np.deg2rad(np.asarray(rpy, dtype=np.float32))
    cr, sr = np.cos(roll), np.sin(roll)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cy, sy = np.cos(yaw), np.sin(yaw)
    rx = np.array(((1, 0, 0), (0, cr, -sr), (0, sr, cr)), dtype=np.float32)
    ry = np.array(((cp, 0, sp), (0, 1, 0), (-sp, 0, cp)), dtype=np.float32)
    rz = np.array(((cy, -sy, 0), (sy, cy, 0), (0, 0, 1)), dtype=np.float32)
    return rz @ ry @ rx


def matrix_to_quaternion_wxyz(matrix: np.ndarray) -> np.ndarray:
    """Convert a proper 3x3 rotation matrix to the pyquaternion convention."""
    matrix = np.asarray(matrix, dtype=np.float64)
    trace = np.trace(matrix)
    if trace > 0:
        scale = 2.0 * np.sqrt(trace + 1.0)
        quat = np.array((0.25 * scale, (matrix[2, 1] - matrix[1, 2]) / scale,
                         (matrix[0, 2] - matrix[2, 0]) / scale,
                         (matrix[1, 0] - matrix[0, 1]) / scale))
    else:
        axis = int(np.argmax(np.diag(matrix)))
        nxt, last = (axis + 1) % 3, (axis + 2) % 3
        scale = 2.0 * np.sqrt(1.0 + matrix[axis, axis] - matrix[nxt, nxt] - matrix[last, last])
        quat = np.zeros(4, dtype=np.float64)
        quat[axis + 1] = 0.25 * scale
        quat[0] = (matrix[last, nxt] - matrix[nxt, last]) / scale
        quat[nxt + 1] = (matrix[nxt, axis] + matrix[axis, nxt]) / scale
        quat[last + 1] = (matrix[last, axis] + matrix[axis, last]) / scale
    return quat.astype(np.float32)


def camera_matrices(sensor: Dict) -> Tuple[np.ndarray, np.ndarray]:
    """Return FarmSim ego-to-optical and homogeneous camera intrinsics."""
    rel = sensor['relative_transform_ue']
    ego_to_camera_ue = np.eye(4, dtype=np.float32)
    ego_to_camera_ue[:3, :3] = rpy_matrix_deg(rel['rotation_deg_rpy'])
    ego_to_camera_ue[:3, 3] = np.asarray(rel['location_cm'], dtype=np.float32) / 100.0
    ue_to_optical = np.array(((0, 1, 0, 0), (0, 0, -1, 0),
                              (1, 0, 0, 0), (0, 0, 0, 1)), dtype=np.float32)
    ego_to_optical = ue_to_optical @ np.linalg.inv(ego_to_camera_ue)
    intr = sensor['intrinsics']
    cam2img = np.eye(4, dtype=np.float32)
    cam2img[0, 0], cam2img[1, 1] = intr['fx'], intr['fy']
    cam2img[0, 2], cam2img[1, 2] = intr['cx'], intr['cy']
    return ego_to_optical, cam2img


def load_frame_cameras(sequence_path: str | Path, frame_id: str) -> Dict[str, Dict]:
    """Build ordered three-camera metadata in the shared optical convention."""
    sequence_path = Path(sequence_path)
    with (sequence_path / 'meta' / f'{frame_id}.json').open('r', encoding='utf-8') as stream:
        meta = json.load(stream)
    sensors = {item['name']: item for item in meta['sensors']
               if item['type'] == 'rgb' and item['enabled']}
    missing = [name for name in RGB_CAMERAS if name not in sensors]
    if missing:
        raise RuntimeError(f'{sequence_path} frame {frame_id} missing cameras: {missing}')
    cameras = {}
    for name in RGB_CAMERAS:
        ego_to_optical, intrinsic = camera_matrices(sensors[name])
        image = next((sequence_path / name / f'{frame_id}{suffix}'
                      for suffix in ('.jpg', '.jpeg', '.png')
                      if (sequence_path / name / f'{frame_id}{suffix}').is_file()), None)
        if image is None:
            raise RuntimeError(f'{sequence_path} frame {frame_id} missing image for {name}')
        cameras[name] = dict(data_path=str(image), lidar2cam=ego_to_optical,
                             cam_intrinsic=intrinsic[:3, :3], cam2img=intrinsic)
    return cameras


def surroundocc_sparse_target(labels: np.ndarray) -> np.ndarray:
    """Convert common dense labels to SurroundOcc's [x,y,z,class] records.

    SurroundOcc reserves 0 for empty voxels.  FarmSim already uses exactly
    that convention, so valid free cells are implicit while the five semantic
    labels remain 1..5.  Ignored cells are written explicitly as 255 so loss
    masking remains correct.
    """
    if labels.shape != OCC_SIZE:
        raise ValueError(f'Expected {OCC_SIZE}, got {labels.shape}')
    keep = labels != 0
    coords = np.argwhere(keep)
    classes = labels[keep].astype(np.int64, copy=True)
    return np.concatenate((coords.astype(np.int64), classes[:, None]), axis=1)


def cotr_labels(labels: np.ndarray) -> np.ndarray:
    """Map common labels to COTR's convention (free is final class 5)."""
    output = labels.copy()
    output[labels == 0] = 5
    output[labels == 1] = 0
    output[labels == 2] = 1
    output[labels == 3] = 2
    output[labels == 4] = 3
    output[labels == 5] = 4
    return output


def cotr_to_common(labels: np.ndarray) -> np.ndarray:
    """Invert :func:`cotr_labels` for the shared FarmSim evaluator."""
    output = labels.copy()
    output[labels == 5] = 0
    output[labels == 0] = 1
    output[labels == 1] = 2
    output[labels == 2] = 3
    output[labels == 3] = 4
    output[labels == 4] = 5
    return output
