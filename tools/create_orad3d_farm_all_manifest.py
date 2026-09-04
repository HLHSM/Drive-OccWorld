#!/usr/bin/env python3
"""Create one ORAD-3D manifest containing every farm/farmland sequence.

The source tree must already be extracted and contain training, validation and
testing directories.  A sequence is selected when one or more VLM scene
descriptions in ``scene_data/*.txt`` contain the whole word ``farm`` or
``farmland``.  The original split is retained in each sequence record.

This is an aggregate analysis manifest, not a held-out test split: it includes
training and validation sequences as explicitly requested.
"""

import argparse
import json
import re
from pathlib import Path


FARM_PATTERN = re.compile(r'\bfarm(?:land)?\b', flags=re.IGNORECASE)
CLASS_NAMES = (
    'free', 'road', 'safe-road', 'car', 'people', 'water', 'snow',
    'grass-on-road', 'rock',
)
SPLITS = ('training', 'validation', 'testing')


def _frame_ids(directory, suffix):
    if not directory.is_dir():
        return set()
    return {path.stem for path in directory.glob(f'*{suffix}') if path.is_file()}


def _sequence_record(sequence_dir, split, min_matching_frames):
    scene_dir = sequence_dir / 'scene_data'
    matches = []
    for path in sorted(scene_dir.glob('*.txt')):
        if FARM_PATTERN.search(path.read_text(encoding='utf-8', errors='replace')):
            matches.append(path.stem)
    if len(matches) < min_matching_frames:
        return None

    frame_ids = sorted(
        _frame_ids(sequence_dir / 'image_data', '.png') &
        _frame_ids(sequence_dir / 'calib', '.txt') &
        _frame_ids(sequence_dir / 'occupancy', '.npy'))
    relative_path = f'{split}/{sequence_dir.name}'
    return dict(
        id=relative_path,
        scenario_id=relative_path,
        path=relative_path,
        original_split=split,
        frame_ids=frame_ids,
        farm_matching_scene_frames=matches,
        farm_matching_scene_frame_count=len(matches),
    )


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-root', type=Path,
                        default=Path('/data/HL/ORAD-3D/extracted'))
    parser.add_argument('--output', type=Path,
                        default=Path('data/orad3d/splits/farm_all.json'))
    parser.add_argument('--min-matching-frames', type=int, default=1)
    return parser.parse_args()


def main():
    args = parse_args()
    if args.min_matching_frames < 1:
        raise ValueError('--min-matching-frames must be at least 1')
    data_root = args.data_root.resolve()
    sequences, excluded_without_labeled_occupancy, summary = [], [], {}
    for split in SPLITS:
        split_root = data_root / split
        if not split_root.is_dir():
            raise FileNotFoundError(f'Missing extracted split directory: {split_root}')
        selected, excluded = [], []
        for sequence_dir in sorted(path for path in split_root.iterdir() if path.is_dir()):
            record = _sequence_record(
                sequence_dir, split, args.min_matching_frames)
            if record is None:
                continue
            if record['frame_ids']:
                selected.append(record)
            else:
                record['exclusion_reason'] = (
                    'no common image_data/calib/occupancy frame')
                excluded.append(record)
        sequences.extend(selected)
        excluded_without_labeled_occupancy.extend(excluded)
        summary[split] = dict(
            all_farm_matching_sequences=len(selected) + len(excluded),
            sequences=len(selected),
            excluded_without_labeled_occupancy=len(excluded),
            labeled_occupancy_frames=sum(len(item['frame_ids']) for item in selected),
            matching_scene_descriptions=sum(
                item['farm_matching_scene_frame_count'] for item in selected),
        )
    if not sequences:
        raise RuntimeError('No farm/farmland sequences were found.')

    manifest = dict(
        dataset='ORAD-3D',
        version=1,
        split='farm-all-original-splits',
        source_root=str(data_root),
        classes=list(CLASS_NAMES),
        selector=dict(
            source='scene_data/*.txt',
            regex=FARM_PATTERN.pattern,
            min_matching_frames=args.min_matching_frames,
            granularity='sequence',
            retains_original_splits=True,
        ),
        summary=summary,
        sequences=sequences,
        farm_matching_sequences_without_labeled_occupancy=
            excluded_without_labeled_occupancy,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('w', encoding='utf-8') as handle:
        json.dump(manifest, handle, indent=2)
        handle.write('\n')

    print(f'wrote {args.output}')
    for split in SPLITS:
        stats = summary[split]
        print(f'{split}: {stats["sequences"]} sequences, '
              f'{stats["labeled_occupancy_frames"]} labeled occupancy frames, '
              f'{stats["matching_scene_descriptions"]} matching descriptions, '
              f'{stats["excluded_without_labeled_occupancy"]} excluded without labels')


if __name__ == '__main__':
    main()
