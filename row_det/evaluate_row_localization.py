#!/usr/bin/env python3
"""Evaluate row identity on common finite support; retain full-extent F1.

This is a new task-specific metric, not a replacement implementation of the
RowDetr benchmark. Short but correctly located rows can match; omissions,
duplicates, wrong orientation and insufficient common support still count.
No detector, prediction, geometric annotation or old score is modified.
"""

import argparse
from collections import defaultdict
import csv
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from row_det.geometric_evaluation import (clip_segment, counts_metrics,
    maximum_matching, point_to_finite_segments)


def straight_segments(rows, bounds):
    """Keep row indices, validating the degree-1 assumption before reducing."""
    result = []
    for index, row in enumerate(rows):
        points = np.asarray(row['points'], dtype=float)[:, :2]
        if len(points) < 2 or not np.isfinite(points).all():
            raise ValueError('Expected a finite row with at least two points')
        length = np.linalg.norm(points[-1] - points[0])
        if length < 1e-10:
            continue
        direction = (points[-1] - points[0]) / length
        projection = (points - points[0]) @ direction
        residual = points - (points[0] + projection[:, None] * direction)
        # Saved detector coordinates are rounded to four decimal places.
        if np.max(np.linalg.norm(residual, axis=1)) > 2e-4 or np.any(np.diff(projection) < -1e-4):
            raise ValueError('Only monotone straight degree-1 row fits are supported')
        segment = clip_segment(points[[0, -1]], bounds)
        if segment is not None:
            result.append((index, segment))
    return result


def score_localization(reference_rows, estimated_rows, xy_range,
                       thresholds=(.1, .2, .3), sample_step=.05,
                       required_fraction=.85, min_common_length=2., angle_limit_deg=3.):
    """One-to-one row localization, with a separate support-completeness score.

    Project both finite rows onto the GT tangent and intersect their supports.
    Require >=2m common support and <=3deg undirected angle. Sample BOTH finite
    segments within that interval at <=0.05m, including its endpoints. At least
    85% of each side's samples must be <delta from the opposite finite segment.
    A GT shorter than 2m remains in the denominator but cannot match by default.
    Unmatched rows, including those without common support, are NOT discarded.
    """
    if (sample_step <= 0 or min_common_length <= 0 or not 0 < required_fraction <= 1
            or not 0 < angle_limit_deg < 90 or any(d <= 0 for d in thresholds)):
        raise ValueError('Invalid localization metric settings')
    refs = straight_segments(reference_rows, xy_range)
    ests = straight_segments(estimated_rows, xy_range)
    shape = (len(refs), len(ests))
    costs = np.full(shape, np.inf)
    candidates = {}
    total_length = sum(np.linalg.norm(s[1] - s[0]) for _, s in refs)
    for i, (_, ref) in enumerate(refs):
        length = float(np.linalg.norm(ref[1] - ref[0]))
        tangent = (ref[1] - ref[0]) / length
        for j, (_, est) in enumerate(ests):
            vector = est[1] - est[0]
            angle = float(np.rad2deg(np.arccos(np.clip(abs(vector @ tangent) / np.linalg.norm(vector), 0, 1))))
            if angle > angle_limit_deg:
                continue
            t = (est - ref[0]) @ tangent
            lo, hi = max(0., min(t)), min(length, max(t))
            overlap = float(hi - lo)
            if overlap < min_common_length - 1e-9:
                continue
            # Parameterizing by the GT tangent ensures identical common support.
            a, b = est[0] + ((np.array([lo, hi]) - t[0]) / (t[1] - t[0]))[:, None] * vector
            n = max(2, int(np.ceil(max(overlap, np.linalg.norm(b - a)) / sample_step)) + 1)
            samples_ref = ref[0] + np.linspace(lo, hi, n)[:, None] * tangent
            samples_est = a + np.linspace(0, 1, n)[:, None] * (b - a)
            d_ref = point_to_finite_segments(samples_ref, [est])
            d_est = point_to_finite_segments(samples_est, [ref])
            costs[i, j] = (d_ref.mean() + d_est.mean()) / 2
            candidates[i, j] = dict(d_ref=d_ref, d_est=d_est, overlap=overlap,
                reference_length=length, angle=angle, sample_count=n)
    results = {}
    for delta in thresholds:
        fractions = {pair: min(float(np.mean(c['d_ref'] < delta)),
                               float(np.mean(c['d_est'] < delta)))
                     for pair, c in candidates.items()}
        eligible = np.zeros(shape, dtype=bool)
        for pair, fraction in fractions.items():
            eligible[pair] = fraction >= required_fraction
        pairs = maximum_matching(eligible, costs)
        matches = []
        for i, j in pairs:
            c = candidates[i, j]
            matches.append(dict(reference_index=refs[i][0], estimate_index=ests[j][0],
                common_support_m=c['overlap'], reference_support_fraction=c['overlap'] / c['reference_length'],
                inlier_fraction=fractions[i, j], mean_distance_m=float(costs[i, j]),
                heading_error_deg=c['angle'], samples_per_side=c['sample_count']))
        result = counts_metrics(len(pairs), len(ests) - len(pairs), len(refs) - len(pairs))
        support = sum(m['common_support_m'] for m in matches)
        result.update(matches=matches, reference_count=len(refs), estimate_count=len(ests),
            matched_common_support_m=support, total_reference_length_m=float(total_length),
            longitudinal_support_coverage=support / total_length if total_length else None)
        results[f'{delta:.2f}'] = result
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('evaluation_dir', type=Path)
    parser.add_argument('output_dir', type=Path)
    parser.add_argument('--thresholds', type=float, nargs='+', default=[.1, .2, .3])
    parser.add_argument('--sample-step', type=float, default=.05)
    parser.add_argument('--required-fraction', type=float, default=.85)
    parser.add_argument('--min-common-length', type=float, default=2.)
    parser.add_argument('--angle-limit-deg', type=float, default=3.)
    args = parser.parse_args()
    paths = sorted(p for p in (args.evaluation_dir / 'row_prediction').glob('*.json') if p.stem[:6].isdigit())
    if not paths:
        raise FileNotFoundError('No saved row predictions')
    args.output_dir.mkdir(parents=True, exist_ok=True)
    frame_dir = args.output_dir / 'per_frame'
    frame_dir.mkdir(exist_ok=True)
    counts = defaultdict(lambda: np.zeros(3, dtype=np.int64))
    support = defaultdict(lambda: np.zeros(2))
    errors = defaultdict(list)
    digest = hashlib.sha256()
    table, cohort, detector, old_protocol = [], [], None, None
    for path in paths:
        raw = path.read_bytes()
        digest.update(path.name.encode() + b'\0' + raw + b'\0')
        item = json.loads(raw)
        if item['config']['curve_degree'] != 1:
            raise ValueError('Only degree-1 fits supported')
        if detector is None:
            detector, old_protocol = item['config'], item['evaluation_protocol']
        if item['config'] != detector or item['evaluation_protocol'] != old_protocol:
            raise ValueError('Mixed detector/evaluation protocols in cohort')
        cohort.append(item['frame'])
        pc = np.asarray(item['point_cloud_range'])
        frame = dict(source_row_file=path.name, frame=item['frame'], localization={}, full_extent={})
        for side, old_key in [('gt', 'label_fit_vs_geometry'), ('pred', 'model_fit_vs_geometry')]:
            frame['localization'][side] = score_localization(item['current']['geometry']['rows'],
                item['current'][side]['rows'], pc[[0, 1, 3, 4]], args.thresholds,
                args.sample_step, args.required_fraction, args.min_common_length, args.angle_limit_deg)
            frame['full_extent'][side] = item['current']['metrics'][old_key]
            for metric, scores in [('localization', frame['localization'][side]),
                                    ('full_extent', frame['full_extent'][side])]:
                for delta, m in scores.items():
                    for group, value in [('overall', 'all'), ('crop_type', item['frame']['crop_type']),
                                         ('sequence', item['frame']['sequence_path'])]:
                        key = (group, value, metric, side, delta)
                        counts[key] += [m['tp'], m['fp'], m['fn']]
                        if metric == 'localization':
                            support[key] += [m['matched_common_support_m'], m['total_reference_length_m']]
                            errors[key].extend(x['mean_distance_m'] for x in m['matches'])
                    table.append(dict(sample_index=item['frame']['sample_index'], crop_type=item['frame']['crop_type'],
                        metric=metric, side=side, delta_m=delta, **{k: m[k] for k in ('tp', 'fp', 'fn', 'f1')}))
        (frame_dir / path.name).write_text(json.dumps(frame, indent=2) + '\n')
    grouped = {}
    for (group, value, metric, side, delta), count in counts.items():
        key = (group, value, metric, side, delta)
        m = counts_metrics(*count)
        if metric == 'localization':
            matched, total = support[key]
            m.update(matched_common_support_m=float(matched), total_reference_length_m=float(total),
                longitudinal_support_coverage=float(matched / total) if total else None,
                matched_pair_mean_distance_m=float(np.mean(errors[key])) if errors[key] else None)
        grouped.setdefault(group, {}).setdefault(value, {}).setdefault(metric, {}).setdefault(side, {})[delta] = m
    protocol = dict(version='common_finite_support_localization_v1', thresholds_m=args.thresholds,
        sample_step_m=args.sample_step, required_inlier_fraction_each_direction=args.required_fraction,
        min_common_support_m=args.min_common_length, angle_limit_deg=args.angle_limit_deg,
        support_axis='geometric_GT_tangent', distance='symmetric_XY_point_to_opposite_finite_segment_on_common_support',
        matching='maximum_cardinality_one_to_one', averaging='micro_pooled_counts',
        reference_filter='same_XY_ROI_and_fit_valid_as_original_no_short_GT_exclusion',
        parameter_status='engineering_defaults_not_selected_on_independent_validation',
        limitations=['Measures row localization on shared support, not complete row reconstruction.',
            'A short row can match a longer GT if common support is at least the fixed minimum.',
            'Longitudinal support coverage includes unmatched GT length in its denominator.',
            'No closed-loop navigation claim; thresholds are not validated vehicle tolerances.'])
    run = json.loads((args.evaluation_dir / 'run_config.json').read_text())
    result = dict(evaluation_dir=str(args.evaluation_dir.resolve()), frames=len(paths),
        sequences=len({f['sequence_path'] for f in cohort}), geometric_rows=sum(f['geometry_row_count'] for f in cohort),
        empty_gt_frames=sum(f['geometry_row_count'] == 0 for f in cohort),
        source_artifacts_sha256=digest.hexdigest(), evaluator_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        checkpoint=run['checkpoint'], camera_geometry=run.get('camera_geometry'),
        protocol=protocol, retained_full_extent_protocol=old_protocol, detector=detector, results=grouped)
    (args.output_dir / 'metrics.json').write_text(json.dumps(result, indent=2) + '\n')
    with (args.output_dir / 'per_frame_metrics.csv').open('w') as file:
        writer = csv.DictWriter(file, fieldnames=list(table[0]))
        writer.writeheader()
        writer.writerows(table)
    print(json.dumps(dict(frames=len(paths), results=grouped['overall']['all']), indent=2))


if __name__ == '__main__':
    main()
