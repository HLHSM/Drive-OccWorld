#!/usr/bin/env python3
"""Cache valid front-grid crop counts for crop-aware FarmSim training sampling."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', required=True)
    parser.add_argument('--data-root', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--target-point-cloud-range', type=float, nargs=6,
                        help='count valid crop voxels only inside this [xmin,ymin,zmin,xmax,ymax,zmax] target grid')
    args = parser.parse_args()

    manifest_path = Path(args.manifest).resolve()
    data_root = Path(args.data_root).resolve()
    output = Path(args.output)
    manifest_bytes = manifest_path.read_bytes()
    digest = hashlib.sha256(manifest_bytes).hexdigest()
    manifest = json.loads(manifest_bytes)
    sequences = manifest['sequences']
    paths = [seq['path'] for seq in sequences]
    target_range = args.target_point_cloud_range
    version = 2 if target_range is not None else 1

    if output.exists():
        cached = json.loads(output.read_text(encoding='utf-8'))
        if (cached.get('version') == version and
                cached.get('target_point_cloud_range') == target_range and
                cached.get('manifest_sha256') == digest and
                cached.get('data_root') == str(data_root) and
                cached.get('sequence_paths') == paths and
                len(cached.get('crop_voxel_counts', [])) == len(sequences) and
                all(len(counts) == len(seq['frame_ids']) for counts, seq in
                    zip(cached['crop_voxel_counts'], sequences))):
            print(f'Reusing {output} ({sum(map(len, cached["crop_voxel_counts"]))} frames)')
            return
        raise ValueError(f'{output} exists but does not match this manifest/data root; '
                         'choose a new output path for the updated index')

    counts_by_sequence = []
    for sequence_index, seq in enumerate(sequences):
        seq_path = data_root / seq['path']
        schema = json.loads((seq_path / 'occupancy_schema.json').read_text())
        region = next((item for item in schema['regions']
                       if item['id'] == schema['region_mode']), None)
        if region is None:
            raise ValueError(f'{seq_path}: missing active occupancy region')
        size_x, size_y, size_z = map(int, region['dimensions_xyz'])
        if target_range is None and size_x % 2:
            raise ValueError(f'{seq_path}: front-only grid needs even x size')
        if target_range is not None:
            source_min = np.asarray(region['min_m_xyz'], dtype=np.float64)
            source_max = np.asarray(region['max_m_xyz'], dtype=np.float64)
            source_shape = np.asarray((size_x, size_y, size_z), dtype=np.int64)
            voxel_size = (source_max - source_min) / source_shape
            target_min = np.asarray(target_range[:3], dtype=np.float64)
            target_max = np.asarray(target_range[3:], dtype=np.float64)
            if np.any(target_min < source_min - 1e-5) or np.any(
                    target_max > source_max + 1e-5):
                raise ValueError(f'{seq_path}: target range exceeds source region')
            target_shape_float = (target_max - target_min) / voxel_size
            target_shape = np.rint(target_shape_float).astype(np.int64)
            if np.any(target_shape < 1) or not np.allclose(
                    target_shape_float, target_shape, atol=1e-4):
                raise ValueError(f'{seq_path}: target range is not a whole '
                                 'number of source voxels')
            target_indices = []
            for axis in range(3):
                centers = target_min[axis] + (
                    np.arange(target_shape[axis], dtype=np.float64) + 0.5) * voxel_size[axis]
                index = np.floor((centers - source_min[axis]) /
                                 voxel_size[axis] + 1e-6).astype(np.int64)
                if index.min() < 0 or index.max() >= source_shape[axis]:
                    raise ValueError(f'{seq_path}: target index outside source grid')
                target_indices.append(index)
        frame_counts = []
        for frame_id in seq['frame_ids']:
            expected = size_x * size_y * size_z
            occupancy = np.fromfile(seq_path / 'occupancy' / f'{frame_id}.bin',
                                    dtype=np.uint8)
            valid = np.fromfile(seq_path / 'occupancy_valid' / f'{frame_id}.bin',
                                dtype=np.uint8)
            if occupancy.size != expected or valid.size != expected:
                raise ValueError(f'{seq_path}/{frame_id}: unexpected occupancy size')
            occupancy = occupancy.reshape(size_z, size_y, size_x)
            valid = valid.reshape(size_z, size_y, size_x)
            if target_range is None:
                occupancy = occupancy[:, :, size_x // 2:]
                valid = valid[:, :, size_x // 2:]
            else:
                x_idx, y_idx, z_idx = target_indices
                occupancy = occupancy[np.ix_(z_idx, y_idx, x_idx)]
                valid = valid[np.ix_(z_idx, y_idx, x_idx)]
            frame_counts.append(int(np.count_nonzero(
                (occupancy == 1) & (valid != 0))))
        counts_by_sequence.append(frame_counts)
        if (sequence_index + 1) % 25 == 0:
            print(f'Scanned {sequence_index + 1}/{len(sequences)} sequences',
                  flush=True)

    output.parent.mkdir(parents=True, exist_ok=True)
    payload = dict(version=version, manifest_sha256=digest,
                   data_root=str(data_root), sequence_paths=paths,
                   crop_voxel_counts=counts_by_sequence)
    if target_range is not None:
        payload['target_point_cloud_range'] = target_range
    temporary = output.with_name(output.name + '.tmp')
    with temporary.open('w', encoding='utf-8') as handle:
        json.dump(payload, handle, separators=(',', ':'))
    temporary.replace(output)
    counts = np.asarray([count for seq in counts_by_sequence for count in seq])
    print(f'Wrote {output}: {len(counts)} frames, '
          f'{np.count_nonzero(counts)} with valid front-grid crop')


if __name__ == '__main__':
    main()
