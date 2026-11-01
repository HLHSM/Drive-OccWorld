#!/usr/bin/env python3
"""Split front3 NPZ captures into route-disjoint, stratified 75/10/15 sets.

All crops, growth stages and captures sharing a route_id stay together.
Optimize frame proportions and crop/stage/lighting/weather distributions.
Whole routes make exact requested proportions generally impossible.
"""

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import zipfile

import numpy as np


SPLITS = ('train', 'val', 'test')
RATIOS = np.array([.75, .10, .15])
FIELDS = ('crop_type', 'growth_stage', 'time_of_day', 'weather')
CAMERAS = ('front_left_rgb', 'front_rgb', 'front_right_rgb')
ROI = [0., -7.5, -2., 15., 7.5, 3.]


def load_sequences(root):
    rows, excluded = [], []
    for seq in sorted(root.glob('*/sequence_*')):
        relative = str(seq.relative_to(root))
        try:
            meta = json.loads((seq / 'scenario_manifest.json').read_text())
            for key in ('scenario_id', 'route_id', *FIELDS):
                if not meta.get(key):
                    raise ValueError(f'missing {key}')
            schema = json.loads((seq / 'occupancy_schema.json').read_text())
            region = next(x for x in schema['regions']
                          if x['id'] == schema['region_mode'])
            low, high = np.array(region['min_m_xyz']), np.array(region['max_m_xyz'])
            if np.any(np.array(ROI[:3]) < low) or np.any(np.array(ROI[3:]) > high):
                raise ValueError('source grid does not cover the target ROI')
            if not np.allclose((high-low)/region['dimensions_xyz'], .2):
                raise ValueError('expected 0.2 m source voxels')
            sets = []
            for camera in CAMERAS:
                sets.append({p.stem for p in (seq / camera).iterdir()
                             if p.suffix.lower() in ('.jpg', '.jpeg', '.png')})
            sets.append({p.stem for p in (seq / 'meta').glob('*.json')})
            sets.append({p.stem for p in (seq / 'occupancy').glob('*.npz')})
            frames = sorted(set.intersection(*sets))
            if not frames:
                raise ValueError('no complete front3/meta/occupancy NPZ frames')
            # Inspect ZIP members without decompressing instance/ground arrays.
            with zipfile.ZipFile(seq / 'occupancy' / f'{frames[0]}.npz') as archive:
                if not {'occupancy.npy', 'occupancy_valid.npy'} <= set(archive.namelist()):
                    raise ValueError('NPZ is missing semantic labels or validity mask')
            rows.append(dict(path=relative, frame_ids=frames, frame_count=len(frames),
                             incomplete_frame_count=len(set.union(*sets))-len(frames),
                             **{key: meta[key] for key in ('scenario_id', 'route_id', *FIELDS)}))
        except (OSError, ValueError, KeyError, StopIteration, zipfile.BadZipFile) as error:
            excluded.append(dict(path=relative, reason=str(error)))
    if not rows:
        raise ValueError(f'No usable sequences found under {root}')
    return rows, excluded


def allocate(rows, seed, restarts=12, iterations=8000):
    """Seeded swap annealing over complete routes with multi-marginal targets."""
    by_route = defaultdict(list)
    for row in rows:
        by_route[row['route_id']].append(row)
    routes = sorted(by_route)
    if len(routes) < 10:
        raise ValueError('At least 10 routes are required for a 10% validation set')
    features = [('frames', None), ('sequences', None)]
    weights = [15000., 10.]
    coverage_indices = []
    for field, weight in zip(FIELDS, (25., 5., 8., 5.)):
        values = sorted({row[field] for row in rows})
        for value in values:
            coverage_indices.append(len(features))
            features.append((field, value))
            weights.append(weight/len(values))
    for fields, weight in ((('crop_type', 'growth_stage'), 8.),
                           (('time_of_day', 'weather'), 5.)):
        values = sorted({tuple(row[f] for f in fields) for row in rows})
        for value in values:
            features.append((fields, value))
            weights.append(weight/len(values))
    vectors = []
    for route in routes:
        group = by_route[route]
        vector = [sum(r['frame_count'] for r in group), len(group)]
        for field, value in features[2:]:
            vector.append(sum(r['frame_count'] for r in group
                              if (tuple(r[f] for f in field) if isinstance(field, tuple)
                                  else r[field]) == value))
        vectors.append(vector)
    vectors = np.array(vectors, dtype=np.float64)
    targets = RATIOS[:, None] * vectors.sum(0)
    scale = np.array(weights)[None, :] / np.maximum(targets, 1.)**2

    def objective(counts):
        # Hardly favor a distribution that omits a crop/stage/time/weather.
        missing = np.count_nonzero(counts[1:, coverage_indices] == 0)
        return float(((counts-targets)**2*scale).sum() + 100.*missing)

    rng = np.random.default_rng(seed)
    # Explore both roundings when 75/15% yields fractional route counts.
    quota_options = []
    n_val = max(1, int(round(len(routes)*RATIOS[1])))
    for n_train in sorted({int(np.floor(len(routes)*RATIOS[0])),
                           int(np.ceil(len(routes)*RATIOS[0]))}):
        quota_options.append([n_train, n_val, len(routes)-n_train-n_val])
    best_score, best_assignment = float('inf'), None
    for restart in range(restarts):
        quotas = quota_options[restart % len(quota_options)]
        assignment = rng.permutation(np.repeat(np.arange(3), quotas))
        counts = np.stack([vectors[assignment == i].sum(0) for i in range(3)])
        score = objective(counts)
        if score < best_score:
            best_score, best_assignment = score, assignment.copy()
        for step in range(iterations):
            a, b = rng.integers(len(routes), size=2)
            i, j = assignment[a], assignment[b]
            if i == j:
                continue
            candidate = counts.copy()
            delta = vectors[b]-vectors[a]
            candidate[i] += delta
            candidate[j] -= delta
            new_score = objective(candidate)
            temperature = .5 * (1.-step/iterations)**3 + .0001
            if new_score < score or rng.random() < np.exp(min(0., (score-new_score)/temperature)):
                counts, score = candidate, new_score
                assignment[a], assignment[b] = j, i
                if score < best_score:
                    best_score, best_assignment = score, assignment.copy()
    route_split = dict(zip(routes, best_assignment.tolist()))
    return {name: [r for r in rows if route_split[r['route_id']] == i]
            for i, name in enumerate(SPLITS)}


def summary(rows):
    result = dict(routes=len({r['route_id'] for r in rows}), sequences=len(rows),
                  frames=sum(r['frame_count'] for r in rows))
    for field in FIELDS:
        result[field] = dict(sorted(Counter(r[field] for r in rows).items()))
        counts = Counter()
        for row in rows:
            counts[row[field]] += row['frame_count']
        result[field+'_frames'] = dict(sorted(counts.items()))
    result['time_weather_frames'] = dict(sorted(Counter({
        f'{time}/{weather}': sum(r['frame_count'] for r in rows
                                 if r['time_of_day'] == time and r['weather'] == weather)
        for time, weather in {(r['time_of_day'], r['weather']) for r in rows}
    }).items()))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-root', type=Path,
                        default=Path('/data/HL/SimData-panoptic/SimData-panoptic'))
    parser.add_argument('--output-dir', type=Path, default=Path('data/simdata_panoptic/splits'))
    parser.add_argument('--seed', type=int, default=20261028)
    parser.add_argument('--reuse-if-current', action='store_true')
    parser.add_argument('--overwrite', action='store_true',
                        help='explicitly regenerate an existing split')
    args = parser.parse_args()
    root = args.data_root.expanduser().resolve()
    rows, excluded = load_sequences(root)
    fingerprint = hashlib.sha256(json.dumps(rows, sort_keys=True).encode()).hexdigest()
    common = dict(source_root=str(root), seed=args.seed,
                  requested_ratios=dict(zip(SPLITS, RATIOS.tolist())),
                  grouping='route_id', inventory_sha256=fingerprint,
                  cameras=list(CAMERAS), point_cloud_range=ROI)
    outputs = [args.output_dir / f'{name}.json' for name in (*SPLITS, 'split_report')]
    if any(p.exists() for p in outputs) and not args.overwrite:
        if args.reuse_if_current and all(p.exists() for p in outputs):
            existing = [json.loads(p.read_text()) for p in outputs]
            if all(all(item.get(k) == v for k, v in common.items()) for item in existing):
                for name, item in zip(SPLITS, existing[:3]):
                    if item.get('split') != name:
                        raise ValueError(f'Incorrect split tag in {name}.json')
                cached_rows = [r for item in existing[:3] for r in item['sequences']]
                if sorted(cached_rows, key=lambda r: r['path']) != rows:
                    raise ValueError('Cached split contents do not match the inventory')
                route_sets = [{r['route_id'] for r in item['sequences']} for item in existing[:3]]
                if any(route_sets[i] & route_sets[j] for i in range(3) for j in range(i)):
                    raise ValueError('Cached manifests have route leakage')
                print(f'Reusing verified route-disjoint splits: {args.output_dir}')
                return
        raise ValueError('Existing split differs or is incomplete; use a new output directory '
                         'or --overwrite to regenerate it explicitly')
    splits = allocate(rows, args.seed)
    report = dict(**common, valid=summary(rows), excluded=excluded,
                  incomplete_frames=sum(r['incomplete_frame_count'] for r in rows),
                  note='Whole routes stay together; ratios are approximate. '
                       'Marginal frame distributions follow the complete dataset.')
    total_frames = report['valid']['frames']
    for name, split_rows in splits.items():
        report[name] = summary(split_rows)
        report[name]['frame_ratio'] = report[name]['frames']/total_frames
        report[name]['missing_categories'] = {
            field: sorted(set(report['valid'][field])-set(report[name][field]))
            for field in FIELDS}
    route_sets = [{r['route_id'] for r in splits[name]} for name in SPLITS]
    assert all(not route_sets[i] & route_sets[j] for i in range(3) for j in range(i))
    report['route_overlap'] = 0
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for name, split_rows in splits.items():
        (args.output_dir / f'{name}.json').write_text(json.dumps(
            dict(**common, split=name, sequences=split_rows), indent=2)+'\n')
    outputs[-1].write_text(json.dumps(report, indent=2)+'\n')
    for name in SPLITS:
        s = report[name]
        print(f"{name}: {s['routes']} routes, {s['sequences']} sequences, "
              f"{s['frames']} frames ({s['frame_ratio']:.3%})")
    print(f'Excluded sequences: {len(excluded)}; incomplete frames: {report["incomplete_frames"]}')
    print(f'Report: {outputs[-1]}')


if __name__ == '__main__':
    main()
