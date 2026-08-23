#!/usr/bin/env python3
"""Create reproducible sequence-level FarmSim train/validation manifests.

The allocator balances crop, growth stage, time of day, weather, and frame
count.  Entire capture sequences stay in one split, preventing temporal and
route leakage between training and validation.
"""
import argparse
import json
import random
from collections import Counter, defaultdict
from pathlib import Path


REQUIRED_DIRS = (
    'front_left_rgb', 'front_rgb', 'front_right_rgb',
    'rear_left_rgb', 'rear_rgb', 'rear_right_rgb',
    'meta', 'occupancy', 'occupancy_valid',
)
RGB_DIRS = set(REQUIRED_DIRS[:6])
RGB_EXTENSIONS = ('.jpg', '.jpeg', '.png')


def common_frame_ids(seq):
    sets = []
    for name in REQUIRED_DIRS:
        if name in RGB_DIRS:
            stems = {
                p.stem for ext in RGB_EXTENSIONS
                for p in (seq / name).glob('*' + ext)
            }
        else:
            suffix = '.json' if name == 'meta' else '.bin'
            stems = {p.stem for p in (seq / name).glob('*' + suffix)}
        sets.append(stems)
    return sorted(set.intersection(*sets))


def load_sequences(root):
    root = root.resolve()
    valid, excluded = [], []
    for seq in sorted(root.glob('*/sequence_*')):
        manifest_path = seq / 'scenario_manifest.json'
        if not manifest_path.is_file() or any(not (seq / x).is_dir() for x in REQUIRED_DIRS):
            excluded.append({'path': str(seq.relative_to(root)), 'reason': 'missing manifest or required directory'})
            continue
        with manifest_path.open(encoding='utf-8') as f:
            meta = json.load(f)
        frames = common_frame_ids(seq)
        if len(frames) < 3:
            excluded.append({'path': str(seq.relative_to(root)), 'reason': 'fewer than 3 complete multi-view frames'})
            continue
        valid.append(dict(path=str(seq.relative_to(root)), frame_ids=frames,
                          scenario_id=meta['scenario_id'], crop_type=meta['crop_type'],
                          growth_stage=meta['growth_stage'], time_of_day=meta['time_of_day'],
                          weather=meta['weather'], frame_count=len(frames)))
    return valid, excluded


def allocate(sequences, val_ratio, seed):
    """Constrained sequence-level multi-marginal stratification.

    Every crop with enough sequences appears in validation. A seeded
    within-crop swap search then balances stage, lighting, weather and frame
    count while preserving the no-sequence-leakage constraint.
    """
    rng = random.Random(seed)
    keys = ('crop_type', 'growth_stage', 'time_of_day', 'weather')
    totals = {key: Counter(s[key] for s in sequences) for key in keys}
    target = {key: {value: count * val_ratio for value, count in counts.items()}
              for key, counts in totals.items()}
    frame_target = sum(s['frame_count'] for s in sequences) * val_ratio
    by_crop = defaultdict(list)
    for seq in sequences:
        by_crop[seq['crop_type']].append(seq)
    selected = set()
    for crop, rows in by_crop.items():
        rng.shuffle(rows)
        quota = max(1, int(round(len(rows) * val_ratio)))
        rows.sort(key=lambda x: x['frame_count'])
        selected.update(x['path'] for x in rows[::max(1, len(rows) // quota)][:quota])

    def objective(paths):
        rows = [s for s in sequences if s['path'] in paths]
        score = ((sum(s['frame_count'] for s in rows) - frame_target) /
                 max(frame_target, 1)) ** 2 * 50
        for key in keys[1:]:
            counts = Counter(s[key] for s in rows)
            for value, expected in target[key].items():
                score += (counts[value] - expected) ** 2
        return score

    score = objective(selected)
    for _ in range(30000):
        crop = rng.choice(list(by_crop))
        inside = [s for s in by_crop[crop] if s['path'] in selected]
        outside = [s for s in by_crop[crop] if s['path'] not in selected]
        if not inside or not outside:
            continue
        remove, add = rng.choice(inside), rng.choice(outside)
        proposal = set(selected)
        proposal.remove(remove['path'])
        proposal.add(add['path'])
        proposal_score = objective(proposal)
        if proposal_score <= score:
            selected, score = proposal, proposal_score
    val_paths = selected
    val = [s for s in sequences if s['path'] in val_paths]
    train = [s for s in sequences if s['path'] not in val_paths]
    return train, val


def summary(items):
    fields = ('crop_type', 'growth_stage', 'time_of_day', 'weather')
    return dict(sequences=len(items), frames=sum(x['frame_count'] for x in items),
                **{field: dict(sorted(Counter(x[field] for x in items).items())) for field in fields})


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-root', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--val-ratio', type=float, default=1 / 6)
    parser.add_argument('--seed', type=int, default=20260821)
    args = parser.parse_args()
    if not 0 < args.val_ratio < 1:
        parser.error('--val-ratio must be between 0 and 1')
    sequences, excluded = load_sequences(args.data_root)
    train, val = allocate(sequences, args.val_ratio, args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    # Paths in the manifests are portable and are resolved by the dataset
    # relative to its explicit ``data_root`` setting.
    common = dict(source_root='.', val_ratio=args.val_ratio, seed=args.seed)
    for name, rows in (('train', train), ('val', val)):
        with (args.output_dir / f'{name}.json').open('w', encoding='utf-8') as f:
            json.dump(dict(**common, split=name, sequences=rows), f, ensure_ascii=False, indent=2)
    report = dict(**common, valid=summary(sequences), train=summary(train), val=summary(val), excluded=excluded)
    with (args.output_dir / 'split_report.json').open('w', encoding='utf-8') as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
