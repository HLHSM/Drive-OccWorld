#!/usr/bin/env python3
"""Create exact-size ORAD-3D current-occupancy smoke-test manifests."""

import argparse
import copy
import json
from pathlib import Path


def make_manifest(source_path, output_path, sample_count):
    with source_path.open(encoding='utf-8') as handle:
        source = json.load(handle)
    candidates = [
        (sequence_index, frame_index)
        for sequence_index, sequence in enumerate(source['sequences'])
        for frame_index in range(len(sequence['frame_ids']))
    ]
    if len(candidates) < sample_count:
        raise RuntimeError(
            f'{source_path} contains {len(candidates)} usable current frames, '
            f'need {sample_count}.')

    # Evenly cover the source ordering while retaining each frame's original
    # sequence path.  ORAD training is current-frame only (queue length zero).
    selected = [candidates[index * len(candidates) // sample_count]
                for index in range(sample_count)]
    frame_ids_by_sequence = {}
    for sequence_index, frame_index in selected:
        frame_ids_by_sequence.setdefault(sequence_index, []).append(
            source['sequences'][sequence_index]['frame_ids'][frame_index])

    result = copy.deepcopy(source)
    result['sequences'] = []
    for sequence_index, frame_ids in frame_ids_by_sequence.items():
        sequence = copy.deepcopy(source['sequences'][sequence_index])
        sequence['frame_ids'] = frame_ids
        result['sequences'].append(sequence)
    result['smoke_test'] = dict(
        source=str(source_path), expected_samples=sample_count,
        queue_length=0,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open('w', encoding='utf-8') as handle:
        json.dump(result, handle, ensure_ascii=False, indent=2)
        handle.write('\n')
    return sum(len(sequence['frame_ids']) for sequence in result['sequences'])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--split-dir', type=Path,
                        default=Path('data/orad3d/splits'))
    parser.add_argument('--output-dir', type=Path,
                        default=Path('data/orad3d/splits/smoke48'))
    parser.add_argument('--train-samples', type=int, default=48)
    parser.add_argument('--test-samples', type=int, default=16)
    args = parser.parse_args()
    if min(args.train_samples, args.test_samples) < 1:
        raise ValueError('sample counts must be positive.')

    train_path = args.output_dir / 'train_48.json'
    test_path = args.output_dir / 'test_16.json'
    train_count = make_manifest(
        args.split_dir / 'train_100.json', train_path, args.train_samples)
    test_count = make_manifest(
        args.split_dir / 'test.json', test_path, args.test_samples)
    print(json.dumps(dict(
        train_manifest=str(train_path), train_samples=train_count,
        test_manifest=str(test_path), test_samples=test_count,
    ), ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
