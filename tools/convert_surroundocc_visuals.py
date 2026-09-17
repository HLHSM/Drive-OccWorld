#!/usr/bin/env python3
"""Convert official SurroundOcc sparse visual outputs to FarmSim NPZ artifacts.

The official visualizer writes non-free voxel centres in metres.  This utility
restores the dense grid, then reuses the exact AgriOcc ground truth and naming
for fair side-by-side rendering with ``render_occ_prediction_previews.py``.
"""

import argparse
import json
from pathlib import Path

import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('manifest', type=Path,
                        help='manifest.json from prepare_qualitative_comparison.py')
    parser.add_argument('reference_prediction_dir', type=Path,
                        help='AgriOcc NPZ artifact directory')
    parser.add_argument('surroundocc_visual_dir', type=Path,
                        help='SURROUNDOCC_VIS_DIR containing <sequence>_<frame>/pred.npy')
    parser.add_argument('output_dir', type=Path,
                        help='output directory for normalized NPZ artifacts')
    return parser.parse_args()


def dense_prediction(vertices, shape, point_cloud_range):
    prediction = np.zeros(shape, dtype=np.uint8)
    if vertices.size == 0:
        return prediction
    if vertices.ndim != 2 or vertices.shape[1] != 4:
        raise ValueError(f'Expected [N, 4] sparse vertices, received {vertices.shape}')
    lower = np.asarray(point_cloud_range[:3], dtype=np.float32)
    upper = np.asarray(point_cloud_range[3:], dtype=np.float32)
    indices = np.rint((vertices[:, :3] - lower) * np.asarray(shape) /
                       (upper - lower) - 0.5).astype(np.int64)
    valid = np.all((indices >= 0) & (indices < np.asarray(shape)), axis=1)
    indices = indices[valid]
    labels = vertices[valid, 3].astype(np.uint8)
    prediction[tuple(indices.T)] = labels
    return prediction


def main():
    args = parse_args()
    rows = json.loads(args.manifest.read_text(encoding='utf-8'))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    written = 0
    for row in rows:
        reference_path = args.reference_prediction_dir / row['reference_npz']
        sequence_name = Path(row['sequence_path']).name
        sparse_path = args.surroundocc_visual_dir / f"{sequence_name}_{row['frame_id']}" / 'pred.npy'
        if not sparse_path.is_file():
            raise FileNotFoundError(f'Missing SurroundOcc visual output: {sparse_path}')
        with np.load(reference_path, allow_pickle=False) as reference:
            gt = reference['current_gt']
            pc_range = reference['point_cloud_range']
            num_classes = int(reference['num_classes'])
            prediction = dense_prediction(np.load(sparse_path), gt.shape, pc_range)
            output_path = args.output_dir / row['reference_npz']
            np.savez_compressed(output_path,
                                sample_index=np.int64(row['sample_index']),
                                current_pred=prediction,
                                current_gt=gt,
                                point_cloud_range=pc_range,
                                num_classes=np.int64(num_classes))
        written += 1
        print(f'[{written}/{len(rows)}] wrote {output_path}', flush=True)
    print(f'Converted {written} aligned SurroundOcc artifacts to {args.output_dir}')


if __name__ == '__main__':
    main()
