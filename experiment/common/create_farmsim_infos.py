#!/usr/bin/env python3
"""Create reproducible external-baseline info pickles from FarmSim splits.

The result contains no copied images or labels.  Each entry references the
original FarmSim files and is restricted to one current frame and three front
RGB cameras, which is the controlled baseline setting for this project.
"""

from __future__ import annotations

import argparse
import pickle
from pathlib import Path

from farmsim_contract import (OCC_SIZE, PC_RANGE, RGB_CAMERAS, load_frame_cameras,
                              read_manifest, resolve_sequence)


def as_pickle_primitives(value):
    """Avoid NumPy objects in infos so every official environment can load them."""
    if hasattr(value, 'tolist'):
        return value.tolist()
    if isinstance(value, dict):
        return {key: as_pickle_primitives(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [as_pickle_primitives(item) for item in value]
    if isinstance(value, list):
        return [as_pickle_primitives(item) for item in value]
    return value


def build_infos(manifest_path: Path, data_root: Path):
    manifest = read_manifest(manifest_path)
    infos = []
    for sequence in manifest['sequences']:
        sequence_path = resolve_sequence(data_root, sequence)
        for frame_id in sequence['frame_ids']:
            cameras = load_frame_cameras(sequence_path, frame_id)
            infos.append(dict(
                token=f"{sequence['scenario_id']}_{sequence_path.name}_{frame_id}",
                scene_token=sequence_path.name,
                frame_id=frame_id,
                sequence_path=str(sequence_path),
                occ_path=str(sequence_path / 'occupancy' / f'{frame_id}.bin'),
                occ_valid_path=str(sequence_path / 'occupancy_valid' / f'{frame_id}.bin'),
                cams=as_pickle_primitives(cameras),
                camera_names=RGB_CAMERAS,
                occ_size=OCC_SIZE,
                pc_range=PC_RANGE,
            ))
    return dict(metadata=dict(dataset='FarmSim', setting='front3_current_occ',
                              split=manifest['split'], source_manifest=str(manifest_path)),
                infos=infos)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--data-root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    payload = build_infos(args.manifest, args.data_root)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('wb') as stream:
        pickle.dump(payload, stream, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"wrote {len(payload['infos'])} current-frame FarmSim infos to {args.output}")


if __name__ == '__main__':
    main()
