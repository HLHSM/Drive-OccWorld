#!/usr/bin/env python3
"""Create a compact ORAD-3D manifest containing only annotated farm frames."""

import argparse
import json
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path,
                        default=Path('data/orad3d/splits/test_farm.json'))
    parser.add_argument('--output', type=Path,
                        default=Path('work_dirs/orad3d_farm_qualitative/farm_frames.json'))
    parser.add_argument('--data-root', type=Path,
                        default=Path('/data/HL/ORAD-3D/extracted'))
    return parser.parse_args()


def main():
    args = parse_args()
    source = json.loads(args.source.read_text(encoding='utf-8'))
    sequences = []
    skipped = []
    for sequence in source['sequences']:
        sequence_path = args.data_root / sequence['path']
        farm_frames = []
        for frame in sequence.get('farm_matching_scene_frames', []):
            required = (
                sequence_path / 'image_data' / f'{frame}.png',
                sequence_path / 'calib' / f'{frame}.txt',
                sequence_path / 'occupancy' / f'{frame}.npy',
            )
            if all(path.is_file() for path in required):
                farm_frames.append(frame)
            else:
                skipped.append(f'{sequence["path"]}/{frame}')
        if not farm_frames:
            continue
        item = {key: value for key, value in sequence.items()
                if key not in ('frame_ids', 'farm_matching_scene_frames',
                               'farm_matching_scene_frame_count')}
        item['frame_ids'] = list(farm_frames)
        sequences.append(item)
    output = {
        key: value for key, value in source.items() if key != 'sequences'
    }
    output.update({
        'split': 'testing-farm-qualitative',
        'description': ('Only the scene-level farm frames identified by '
                        'test_farm.json; used for qualitative comparison.'),
        'sequences': sequences,
    })
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2) + '\n', encoding='utf-8')
    print(f'Wrote {args.output}: {len(sequences)} sequences, '
          f'{sum(len(item["frame_ids"]) for item in sequences)} farm frames')
    if skipped:
        print(f'Skipped {len(skipped)} farm frames without complete '
              f'RGB/calibration/occupancy data')


if __name__ == '__main__':
    main()
