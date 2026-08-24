#!/usr/bin/env python3
"""Create tiny, sequence-safe FarmSim manifests for training smoke tests."""

import argparse
import copy
import json
from pathlib import Path


def make_manifest(source_path, output_path, sample_count, queue_length):
    with source_path.open(encoding='utf-8') as file:
        manifest = json.load(file)

    # The FarmSim dataset creates one sample for each index in
    # ``range(queue_length, len(frame_ids))`` when no future target is used.
    # Truncating one sequence therefore retains temporal ordering and gives an
    # exact, deterministic number of dataset samples.
    for sequence in manifest['sequences']:
        if len(sequence['frame_ids']) >= queue_length + sample_count:
            result = copy.deepcopy(manifest)
            row = copy.deepcopy(sequence)
            row['frame_ids'] = row['frame_ids'][:queue_length + sample_count]
            result['sequences'] = [row]
            result['smoke_test'] = dict(
                source=str(source_path),
                queue_length=queue_length,
                expected_samples=sample_count,
            )
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with output_path.open('w', encoding='utf-8') as file:
                json.dump(result, file, ensure_ascii=False, indent=2)
                file.write('\n')
            return row['path']
    raise RuntimeError(
        f'No sequence in {source_path} has at least '
        f'{queue_length + sample_count} frames.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--split-dir', type=Path,
                        default=Path('data/farmsim/splits'))
    parser.add_argument('--output-dir', type=Path,
                        default=Path('data/farmsim/splits/smoke48'))
    parser.add_argument('--train-samples', type=int, default=48)
    parser.add_argument('--val-samples', type=int, default=16)
    parser.add_argument('--queue-length', type=int, default=2)
    args = parser.parse_args()
    if min(args.train_samples, args.val_samples, args.queue_length) < 1:
        raise ValueError('sample counts and queue length must be positive.')

    train_path = args.output_dir / 'train_48.json'
    val_path = args.output_dir / 'val_16.json'
    train_sequence = make_manifest(
        args.split_dir / 'train.json', train_path,
        args.train_samples, args.queue_length)
    val_sequence = make_manifest(
        args.split_dir / 'val.json', val_path,
        args.val_samples, args.queue_length)
    print(json.dumps(dict(
        train_manifest=str(train_path), train_samples=args.train_samples,
        train_sequence=train_sequence,
        val_manifest=str(val_path), val_samples=args.val_samples,
        val_sequence=val_sequence,
    ), ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
