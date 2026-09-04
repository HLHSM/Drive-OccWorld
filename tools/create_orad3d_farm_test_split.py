#!/usr/bin/env python3
"""Create a sequence-safe ORAD-3D test subset from farm scene descriptions.

The official archive names do not carry terrain labels.  ORAD-3D's VLM
``scene_data/*.txt`` descriptions are therefore used as the selection source.
An archive is included when at least ``--min-matching-frames`` descriptions
contain the whole word ``farm`` or ``farmland``.  Frame ids are obtained from
the image/calibration/occupancy intersection in that same archive, so the
result can be consumed directly by ``ORAD3DWorldDataset`` after extraction.
"""

import argparse
import json
import re
import zipfile
from pathlib import Path


FARM_PATTERN = re.compile(r'\bfarm(?:land)?\b', flags=re.IGNORECASE)
CLASS_NAMES = (
    'free', 'road', 'safe-road', 'car', 'people', 'water', 'snow',
    'grass-on-road', 'rock',
)


def _frame_ids(names, directory, suffix):
    prefix = f'{directory}/'
    return {
        name[len(prefix):-len(suffix)]
        for name in names
        if name.startswith(prefix) and name.endswith(suffix)
    }


def _sequence_from_archive(archive, min_matching_frames):
    with zipfile.ZipFile(archive) as handle:
        names = handle.namelist()
        images = _frame_ids(names, 'image_data', '.png')
        calibrations = _frame_ids(names, 'calib', '.txt')
        occupancy = _frame_ids(names, 'occupancy', '.npy')
        frame_ids = sorted(images & calibrations & occupancy)
        matches = []
        for name in names:
            if not (name.startswith('scene_data/') and name.endswith('.txt')):
                continue
            text = handle.read(name).decode('utf-8', errors='replace')
            if FARM_PATTERN.search(text):
                matches.append(Path(name).stem)

    if len(matches) < min_matching_frames or not frame_ids:
        return None
    sequence_id = archive.stem
    return dict(
        id=f'testing/{sequence_id}',
        scenario_id=f'testing/{sequence_id}',
        path=f'testing/{sequence_id}',
        frame_ids=frame_ids,
        farm_matching_scene_frames=sorted(matches),
        farm_matching_scene_frame_count=len(matches),
    )


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--zip-dir', type=Path,
                        default=Path('/data/HL/ORAD-3D/testing_zip'))
    parser.add_argument('--source-root', type=Path,
                        default=Path('/data/HL/ORAD-3D/extracted'),
                        help='root used after archives are extracted')
    parser.add_argument('--output', type=Path,
                        default=Path('data/orad3d/splits/test_farm.json'))
    parser.add_argument('--min-matching-frames', type=int, default=1)
    return parser.parse_args()


def main():
    args = parse_args()
    if args.min_matching_frames < 1:
        raise ValueError('--min-matching-frames must be at least 1')
    archives = sorted(args.zip_dir.glob('*.zip'))
    if not archives:
        raise FileNotFoundError(f'No .zip archives found in {args.zip_dir}')

    sequences = []
    for archive in archives:
        sequence = _sequence_from_archive(archive, args.min_matching_frames)
        if sequence is not None:
            sequences.append(sequence)
    if not sequences:
        raise RuntimeError(
            f'No farm/farmland test archives met {args.min_matching_frames} matching frames')

    manifest = dict(
        dataset='ORAD-3D',
        version=1,
        split='testing-farm',
        source_root=str(args.source_root.resolve()),
        classes=list(CLASS_NAMES),
        selector=dict(
            source='scene_data/*.txt',
            regex=FARM_PATTERN.pattern,
            min_matching_frames=args.min_matching_frames,
            granularity='sequence',
        ),
        sequences=sequences,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('w', encoding='utf-8') as handle:
        json.dump(manifest, handle, indent=2)
        handle.write('\n')

    frame_count = sum(len(sequence['frame_ids']) for sequence in sequences)
    evidence_count = sum(sequence['farm_matching_scene_frame_count']
                         for sequence in sequences)
    print(f'wrote {args.output}')
    print(f'farm test sequences: {len(sequences)}')
    print(f'labeled occupancy frames: {frame_count}')
    print(f'matching scene descriptions: {evidence_count}')


if __name__ == '__main__':
    main()
