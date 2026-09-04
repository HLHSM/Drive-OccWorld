#!/usr/bin/env python3
"""Build deterministic ORAD-3D occupancy manifests from extracted sequences.

Expected layout::

    <data-root>/training/<sequence>/{image_data,calib,occupancy}/
    <data-root>/validation/<sequence>/...
    <data-root>/testing/<sequence>/...

Only frames with all three current-frame inputs are included.  Training
fractions are nested sequence-level subsets so no neighbouring frames leak
between a small-data train subset and a held-out sequence.
"""

import argparse
import hashlib
import json
import math
from collections import Counter
from pathlib import Path

import numpy as np


CLASS_NAMES = (
    'free', 'road', 'safe-road', 'car', 'people', 'water', 'snow',
    'grass-on-road', 'rock',
)


def _frame_ids(directory, suffix, trim_suffix=''):
    if not directory.is_dir():
        return set()
    return {
        path.name[:-len(suffix)] + trim_suffix
        for path in directory.iterdir()
        if path.is_file() and path.name.endswith(suffix)
    }


def _occupancy_histogram(paths):
    histogram = Counter()
    for path in paths:
        sparse = np.load(path, allow_pickle=False)
        if sparse.ndim != 2 or sparse.shape[1] != 4:
            raise ValueError(f'{path}: expected occupancy shape [N, 4], got {sparse.shape}.')
        labels = sparse[:, 3]
        if not np.issubdtype(labels.dtype, np.integer):
            if not np.equal(labels, np.rint(labels)).all():
                raise ValueError(f'{path}: occupancy labels must be integers.')
            labels = np.rint(labels).astype(np.int64)
        labels = labels.astype(np.int64, copy=False)
        invalid = labels[(labels < 0) | (labels >= len(CLASS_NAMES))]
        if invalid.size:
            raise ValueError(f'{path}: unsupported class ids {np.unique(invalid).tolist()}.')
        histogram.update(labels.tolist())
    return histogram


def _discover_split(data_root, split, audit_labels):
    split_root = data_root / split
    if not split_root.is_dir():
        raise FileNotFoundError(
            f'Missing {split_root}. Extract ORAD archives first; expected '
            f'{data_root}/{{training,validation,testing}}/<sequence>.')
    sequences = []
    for sequence_dir in sorted(path for path in split_root.iterdir() if path.is_dir()):
        images = _frame_ids(sequence_dir / 'image_data', '.png')
        calibrations = _frame_ids(sequence_dir / 'calib', '.txt')
        occupancies = _frame_ids(sequence_dir / 'occupancy', '.npy')
        frame_ids = sorted(images & calibrations & occupancies)
        if not frame_ids:
            continue
        occupancy_paths = [sequence_dir / 'occupancy' / f'{frame_id}.npy'
                           for frame_id in frame_ids]
        histogram = _occupancy_histogram(occupancy_paths) if audit_labels else Counter()
        sequences.append(dict(
            id=f'{split}/{sequence_dir.name}',
            scenario_id=f'{split}/{sequence_dir.name}',
            path=f'{split}/{sequence_dir.name}',
            frame_ids=frame_ids,
            class_histogram={str(label): int(histogram[label])
                             for label in range(len(CLASS_NAMES))
                             if histogram[label]},
        ))
    if not sequences:
        raise RuntimeError(f'No labeled ORAD occupancy frames found under {split_root}.')
    return sequences


def _stable_seed(seed, sequence_id):
    digest = hashlib.sha256(f'{seed}:{sequence_id}'.encode('utf-8')).digest()
    return int.from_bytes(digest[:8], 'little')


def _nested_sequence_order(sequences, seed):
    """Prioritize rare-class coverage, with a deterministic tie-breaker."""
    total = Counter()
    for sequence in sequences:
        total.update({int(key): value
                      for key, value in sequence['class_histogram'].items()})
    rarity = {label: 1.0 / max(total[label], 1) for label in range(len(CLASS_NAMES))}

    def score(sequence):
        histogram = {int(key): value
                     for key, value in sequence['class_histogram'].items()}
        # Do not let the very common free class dominate subset selection.
        semantic_coverage = sum(rarity[label] for label in histogram if label != 0)
        labeled_frames = len(sequence['frame_ids'])
        return (-semantic_coverage, -min(labeled_frames, 512),
                _stable_seed(seed, sequence['id']))

    return sorted(sequences, key=score)


def _prefix_for_fraction(ordered_sequences, fraction):
    total_frames = sum(len(sequence['frame_ids']) for sequence in ordered_sequences)
    target_frames = int(math.ceil(total_frames * fraction))
    selected, selected_frames = [], 0
    for sequence in ordered_sequences:
        if selected and selected_frames >= target_frames:
            break
        selected.append(sequence)
        selected_frames += len(sequence['frame_ids'])
    return selected, selected_frames, total_frames


def _write_manifest(path, data_root, split, sequences, seed, requested_fraction=None,
                    total_train_frames=None):
    manifest = dict(
        dataset='ORAD-3D',
        version=1,
        source_root=str(data_root),
        split=split,
        seed=seed,
        classes=list(CLASS_NAMES),
        requested_train_fraction=requested_fraction,
        total_train_frames=total_train_frames,
        sequences=sequences,
    )
    with path.open('w', encoding='utf-8') as handle:
        json.dump(manifest, handle, indent=2)
        handle.write('\n')


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-root', type=Path,
                        default=Path('/data/HL/ORAD-3D/extracted'))
    parser.add_argument('--output-dir', type=Path,
                        default=Path('data/orad3d/splits'))
    parser.add_argument('--seed', type=int, default=20260904)
    parser.add_argument('--fractions', type=float, nargs='+',
                        default=(0.10, 0.25, 0.50, 1.00))
    parser.add_argument('--skip-label-audit', action='store_true',
                        help='skip reading every .npy during manifest creation')
    return parser.parse_args()


def main():
    args = parse_args()
    fractions = sorted(set(args.fractions))
    if not fractions or any(not 0 < fraction <= 1 for fraction in fractions):
        raise ValueError('--fractions must be in (0, 1].')
    data_root = args.data_root.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    train = _discover_split(data_root, 'training', not args.skip_label_audit)
    validation = _discover_split(data_root, 'validation', not args.skip_label_audit)
    testing = _discover_split(data_root, 'testing', not args.skip_label_audit)
    ordered_train = _nested_sequence_order(train, args.seed)
    total_train_frames = sum(len(sequence['frame_ids']) for sequence in train)

    _write_manifest(output_dir / 'val.json', data_root, 'validation', validation,
                    args.seed)
    _write_manifest(output_dir / 'test.json', data_root, 'testing', testing,
                    args.seed)

    subset_stats = {}
    for fraction in fractions:
        selected, selected_frames, total = _prefix_for_fraction(ordered_train, fraction)
        name = f'train_{int(round(fraction * 100)):03d}.json'
        _write_manifest(output_dir / name, data_root, 'training', selected,
                        args.seed, requested_fraction=fraction,
                        total_train_frames=total_train_frames)
        subset_stats[f'{int(round(fraction * 100))}%'] = dict(
            manifest=name,
            sequences=len(selected),
            frames=selected_frames,
            actual_fraction=selected_frames / total if total else 0.0,
        )

    stats = dict(
        data_root=str(data_root), seed=args.seed, classes=list(CLASS_NAMES),
        audit_labels=not args.skip_label_audit,
        splits=dict(
            training=dict(sequences=len(train), frames=total_train_frames),
            validation=dict(sequences=len(validation),
                            frames=sum(len(sequence['frame_ids']) for sequence in validation)),
            testing=dict(sequences=len(testing),
                         frames=sum(len(sequence['frame_ids']) for sequence in testing)),
        ),
        train_subsets=subset_stats,
    )
    with (output_dir / 'stats.json').open('w', encoding='utf-8') as handle:
        json.dump(stats, handle, indent=2)
        handle.write('\n')
    print(json.dumps(stats, indent=2))


if __name__ == '__main__':
    main()
