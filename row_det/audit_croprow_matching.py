#!/usr/bin/env python3
"""Independently audit saved finite-row scores; never change official metrics.

Uses GEOS/Shapely clipping and distances and SciPy bipartite matching. Infinite
supporting lines are a counterfactual diagnostic ONLY, not eligible navigation
paths, metric replacements, or evidence that missing endpoints were observed.
Run with a Python environment containing numpy, scipy and shapely >= 2.
"""

import argparse
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
import csv
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import scipy
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import maximum_bipartite_matching
from scipy.spatial.transform import Rotation
import shapely
from shapely.geometry import LineString, MultiLineString, box

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from row_det.geometric_evaluation import counts_metrics, point_to_finite_segments


def line_parts(geometry):
    if geometry.geom_type == 'LineString':
        return [geometry] if geometry.length > 1e-10 else []
    if hasattr(geometry, 'geoms'):
        return [part for child in geometry.geoms for part in line_parts(child)]
    return []


def clipped_rows(rows, bounds):
    result = []
    for index, row in enumerate(rows):
        line = LineString(np.asarray(row['points'])[:, :2])
        parts = line_parts(line.intersection(box(*bounds)))
        if parts:
            result.append((index, MultiLineString(parts), parts))
    return result


def independent_samples(parts, step):
    lengths = np.array([part.length for part in parts])
    total = lengths.sum()
    locations = np.linspace(0, total, max(2, int(np.ceil(total / step)) + 1))
    # Geometry references are single clipped straight segments. The general
    # multipart case samples actual parts without bridging a missing interval.
    ends = np.cumsum(lengths)
    indices = np.minimum(np.searchsorted(ends, locations, side='right'), len(parts) - 1)
    starts = np.r_[0., ends[:-1]]
    points = [parts[i].interpolate(float(t - starts[i])) for t, i in zip(locations, indices)]
    return shapely.get_coordinates(points), np.array(points, dtype=object)


def cardinality(eligible):
    if not all(eligible.shape):
        return 0
    return int(np.sum(maximum_bipartite_matching(csr_matrix(eligible), perm_type='column') >= 0))


def d85(distances, required):
    if not distances.shape[1]:
        return np.empty(0)
    k = int(np.ceil(required * len(distances))) - 1
    return np.partition(distances, k, axis=0)[k]


def audit_frame(path):
    artifact = json.loads(Path(path).read_text())
    current = artifact['current']
    protocol = artifact['evaluation_protocol']
    thresholds = protocol['thresholds_m']
    required = protocol['required_coverage']
    pc = np.asarray(artifact['point_cloud_range'])
    bounds = pc[[0, 1, 3, 4]]
    reference = clipped_rows(current['geometry']['rows'], bounds)
    # Check the saved world-to-grid transform via SciPy's independent inverse
    # rotation and GEOS clipping. This does not reannotate or filter truth.
    grid = artifact['geometry_diagnostics']['grid_frame_world_ue']
    if not np.allclose(grid['rotation_deg_rpy'][:2], 0, atol=1e-6):
        raise ValueError('World XY is insufficient for a nonlevel grid')
    inverse_yaw = Rotation.from_euler('z', grid['rotation_deg_rpy'][2], degrees=True).inv()
    maximum_transform_difference = 0.
    for row in current['geometry']['rows']:
        delta = np.array(row['world_points']) - np.array(grid['location_cm'][:2]) / 100
        local = inverse_yaw.apply(np.column_stack([delta, np.zeros(len(delta))]))[:, :2]
        independent = LineString(local).intersection(box(*bounds))
        difference = float(independent.hausdorff_distance(LineString(row['points'])))
        if not np.isfinite(difference):
            raise ValueError('Saved geometric row has no independently clipped world support')
        maximum_transform_difference = max(maximum_transform_difference, difference)
    totals, details, errors, diagnostic = {}, [], [], {}
    max_distance_difference = 0.
    pair_count = 0
    for side, comparison in [('gt', 'label_fit_vs_geometry'), ('pred', 'model_fit_vs_geometry')]:
        estimated = clipped_rows(current[side]['rows'], bounds)
        distances, infinite, samples_per_row = [], [], []
        for ref_index, _, parts in reference:
            coords, points = independent_samples(parts, protocol['sample_step_m'])
            samples_per_row.append(len(points))
            matrix = np.empty((len(points), len(estimated)))
            unlimited = np.empty_like(matrix)
            for j, (_, geometry, est_parts) in enumerate(estimated):
                matrix[:, j] = shapely.distance(points, geometry)
                primitive = np.concatenate([np.stack([np.asarray(p.coords)[:-1],
                                                     np.asarray(p.coords)[1:]], axis=1)
                                            for p in est_parts])
                # Check the production finite distance primitive independently
                # at up to 11 positions on EVERY reference/estimate pair.
                chosen = np.unique(np.linspace(0, len(points) - 1, min(11, len(points))).astype(int))
                actual = point_to_finite_segments(coords[chosen], primitive)
                max_distance_difference = max(max_distance_difference,
                    float(np.max(np.abs(actual - matrix[chosen, j]))))
                pair_count += 1
                all_xy = np.concatenate([np.asarray(p.coords) for p in est_parts])
                origin, endpoint = all_xy[0], all_xy[-1]
                vector = endpoint - origin
                normal = np.array([-vector[1], vector[0]]) / np.linalg.norm(vector)
                # Saved display points are rounded to four decimals. Allow
                # their sub-millimetre quantization, never a curved fit.
                if artifact['config']['curve_degree'] != 1 or np.max(np.abs((all_xy - origin) @ normal)) > 1e-3:
                    raise ValueError('Infinite-line diagnostic requires degree-1 straight fits')
                unlimited[:, j] = np.abs((coords - origin) @ normal)
            distances.append(matrix)
            infinite.append(unlimited)
        for threshold in thresholds:
            key = f'{threshold:.2f}'
            eligible = np.array([np.mean(d < threshold, axis=0) >= required for d in distances],
                                dtype=bool).reshape(len(reference), len(estimated))
            infinite_eligible = np.array([np.mean(d < threshold, axis=0) >= required for d in infinite],
                                         dtype=bool).reshape(eligible.shape)
            tp = cardinality(eligible)
            counts = [tp, len(estimated) - tp, len(reference) - tp]
            saved = current['metrics'][comparison][key]
            if counts != [saved['tp'], saved['fp'], saved['fn']]:
                errors.append(dict(file=Path(path).name, comparison=comparison, threshold=key,
                                   saved=[saved['tp'], saved['fp'], saved['fn']], independent=counts))
            ref_lookup = {original: i for i, (original, _, _) in enumerate(reference)}
            est_lookup = {original: j for j, (original, _, _) in enumerate(estimated)}
            seen_ref, seen_est = set(), set()
            for match in saved['matches']:
                i, j = ref_lookup[match['reference_index']], est_lookup[match['estimate_index']]
                coverage = float(np.mean(distances[i][:, j] < threshold))
                if (i in seen_ref or j in seen_est or not eligible[i, j]
                        or abs(coverage - match['coverage']) > 1e-10
                        or abs(float(distances[i][:, j].mean()) - match['mean_distance_m']) > 1e-9
                        or samples_per_row[i] != match['sample_count']):
                    errors.append(dict(file=Path(path).name, comparison=comparison,
                                       threshold=key, invalid_saved_match=match))
                seen_ref.add(i)
                seen_est.add(j)
            if len(seen_ref) != tp:
                errors.append(dict(file=Path(path).name, comparison=comparison,
                                   threshold=key, saved_pairs=len(seen_ref), independent_tp=tp))
            totals[(comparison, key)] = counts
            # Fixed 0.20m diagnostic; the official protocol is never changed.
            if key != '0.20':
                continue
            reasons = Counter()
            for i, (original_index, ref_geometry, _) in enumerate(reference):
                if i in seen_ref:
                    reason = 'matched'
                elif not len(estimated):
                    reason = 'no_estimates'
                elif eligible[i].any():
                    reason = 'one_to_one_competition'
                elif infinite_eligible[i].any():
                    reason = 'endpoint_or_extent_limited_candidate'
                else:
                    reason = 'lateral_heading_or_missing'
                reasons[reason] += 1
                finite_d85 = d85(distances[i], required)
                infinite_d85 = d85(infinite[i], required)
                item = dict(file=Path(path).name, sample_index=artifact['frame']['sample_index'],
                            crop_type=artifact['frame']['crop_type'], comparison=comparison,
                            reference_index=original_index, reference_length_m=ref_geometry.length,
                            samples=samples_per_row[i], classification=reason,
                            best_estimate_index='', best_finite_d85_m='', best_infinite_d85_m='',
                            best_finite_coverage='', angle_error_deg='', midpoint_lateral_error_m='',
                            estimate_length_m='')
                endpoints = np.array(current['geometry']['rows'][original_index]['points'])[:, :2]
                direction = endpoints[-1] - endpoints[0]
                item['reference_angle_deg'] = float((np.degrees(np.arctan2(direction[1], direction[0])) + 90) % 180 - 90)
                item['shorter_than_detector_minimum'] = ref_geometry.length < artifact['config']['min_row_length']
                if len(estimated):
                    j = int(np.argmin(finite_d85))
                    a = np.array(current['geometry']['rows'][original_index]['points'])[:, :2]
                    b = np.asarray(estimated[j][1].geoms[0].coords)
                    va, vb = a[-1] - a[0], b[-1] - b[0]
                    cosine = np.clip(abs(va @ vb) / (np.linalg.norm(va) * np.linalg.norm(vb)), 0, 1)
                    normal = np.array([-vb[1], vb[0]]) / np.linalg.norm(vb)
                    item.update(best_estimate_index=estimated[j][0],
                                best_finite_d85_m=float(finite_d85[j]),
                                best_infinite_d85_m=float(infinite_d85.min()),
                                best_finite_coverage=float(np.mean(distances[i][:, j] < threshold)),
                                angle_error_deg=float(np.degrees(np.arccos(cosine))),
                                midpoint_lateral_error_m=float(abs((a.mean(axis=0) - b[0]) @ normal)),
                                estimate_length_m=estimated[j][1].length)
                details.append(item)
            diagnostic[comparison] = dict(reasons=dict(reasons),
                infinite_line_counterfactual_tp=cardinality(infinite_eligible))
    return dict(totals=totals, details=details, errors=errors, diagnostic=diagnostic,
                max_distance_difference_m=max_distance_difference, distance_pairs_checked=pair_count,
                max_world_to_grid_difference_m=maximum_transform_difference)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--evaluation-dir', action='append', required=True, type=Path)
    parser.add_argument('--output-dir', required=True, type=Path)
    parser.add_argument('--workers', type=int, default=8)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    production_path = Path(__file__).with_name('geometric_evaluation.py')
    summary = dict(versions=dict(numpy=np.__version__, scipy=scipy.__version__, shapely=shapely.__version__),
                   audit_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                   production_metric_sha256=hashlib.sha256(production_path.read_bytes()).hexdigest(), evaluations={})
    report = ['# Independent finite-row matching audit', '',
              'GEOS clipping/distance + SciPy maximum bipartite matching, against saved metrics. '
              'Official predictions, fits and scores are unchanged.', '',
              '| Evaluation | Side | Frames | TP@0.20m | F1@0.20m | Mismatches |',
              '| --- | --- | ---: | ---: | ---: | ---: |']
    findings = []
    for root in args.evaluation_dir:
        paths = sorted(path for path in (root / 'row_prediction').glob('*.json') if path.stem[:6].isdigit())
        if not paths:
            raise ValueError(f'No frame row predictions in {root}')
        counts = defaultdict(lambda: np.zeros(3, dtype=np.int64))
        reasons, unlimited_tp = defaultdict(Counter), Counter()
        errors, all_details = [], []
        maximum_difference, maximum_transform_difference, pair_count = 0., 0., 0
        source_digest = hashlib.sha256()
        for path in paths:
            source_digest.update(path.name.encode() + b'\0' + path.read_bytes() + b'\0')
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            for n, result in enumerate(pool.map(audit_frame, map(str, paths), chunksize=4), 1):
                for key, value in result['totals'].items():
                    counts[key] += value
                for comparison, value in result['diagnostic'].items():
                    reasons[comparison].update(value['reasons'])
                    unlimited_tp[comparison] += value['infinite_line_counterfactual_tp']
                errors.extend(result['errors'])
                all_details.extend(result['details'])
                maximum_difference = max(maximum_difference, result['max_distance_difference_m'])
                maximum_transform_difference = max(maximum_transform_difference, result['max_world_to_grid_difference_m'])
                pair_count += result['distance_pairs_checked']
                if n % 100 == 0 or n == len(paths):
                    print(f'{root.name}: {n}/{len(paths)}; mismatches={len(errors)}', flush=True)
        aggregate = {}
        for (comparison, key), value in counts.items():
            aggregate.setdefault(comparison, {})[key] = counts_metrics(*value)
        destination = args.output_dir / root.name
        destination.mkdir(exist_ok=True)
        with (destination / 'reference_diagnostics_0p20m.csv').open('w') as handle:
            writer = csv.DictWriter(handle, fieldnames=list(all_details[0]))
            writer.writeheader()
            writer.writerows(all_details)
        evaluation = dict(source=str(root.resolve()), frames=len(paths), metrics=aggregate,
                          ordered_row_artifacts_sha256=source_digest.hexdigest(),
                          mismatches=errors, distance_pairs_checked=pair_count,
                          max_distance_difference_m=maximum_difference,
                          max_world_to_grid_difference_m=maximum_transform_difference,
                          false_negative_diagnostics={key: dict(value) for key, value in reasons.items()},
                          infinite_line_counterfactual_tp=dict(unlimited_tp))
        evaluation['candidate_geometry_statistics'] = {}
        for comparison in aggregate:
            records = [item for item in all_details if item['comparison'] == comparison]
            statistics = dict(reference_rows=len(records),
                references_shorter_than_detector_minimum=sum(item['shorter_than_detector_minimum'] for item in records),
                references_outside_detector_angle_search=sum(abs(item['reference_angle_deg']) > 60 for item in records))
            for field in ['angle_error_deg', 'midpoint_lateral_error_m', 'best_finite_d85_m']:
                values = [item[field] for item in records if item[field] != '']
                statistics[field] = dict(count=len(values), quantiles_25_50_75_90=
                    np.quantile(values, [.25, .5, .75, .9]).tolist() if values else [])
            evaluation['candidate_geometry_statistics'][comparison] = statistics
        summary['evaluations'][root.name] = evaluation
        for comparison in aggregate:
            metric = aggregate[comparison]['0.20']
            report.append(f"| {root.name} | {comparison} | {len(paths)} | {metric['tp']} | "
                          f"{metric['f1']:.6f} | {len(errors)} |")
        findings += ['', f'## {root.name}', '',
                   f'Checked {pair_count} reference-estimate pairs at up to 11 positions each; '
                   f'maximum NumPy-versus-GEOS distance difference: {maximum_difference:.3g} m.', '',
                   f'Maximum SciPy world-to-grid versus saved clipped geometry difference: '
                   f'{maximum_transform_difference:.3g} m.', '',
                   '| Side | Matched | No estimates | One-to-one competition | Extent-limited candidate | Lateral/heading/missing |',
                   '| --- | ---: | ---: | ---: | ---: | ---: |']
        for comparison, value in reasons.items():
            keys = ['matched', 'no_estimates', 'one_to_one_competition',
                    'endpoint_or_extent_limited_candidate', 'lateral_heading_or_missing']
            findings.append('| ' + comparison + ' | ' + ' | '.join(str(value.get(k, 0)) for k in keys) + ' |')
        findings += ['', 'Extent-limited means at least one infinite supporting line passes the 85% gate '
                   'while no finite estimate passes. This is a candidate-based counterfactual; it does '
                   'not prove correspondence or that all such rows are recoverable one-to-one. '
                   'Infinite-line counts are diagnostics only. Lateral/heading/missing does not '
                   'distinguish an absent row from an incorrectly positioned row.', '']
    summary['passed'] = all(not item['mismatches'] and item['max_distance_difference_m'] < 1e-9
                            and item['max_world_to_grid_difference_m'] < 1e-8
                            for item in summary['evaluations'].values())
    (args.output_dir / 'matching_audit.json').write_text(json.dumps(summary, indent=2) + '\n')
    (args.output_dir / 'REPORT.md').write_text('\n'.join(report + findings))
    print(f'passed={summary["passed"]}; output={args.output_dir}', flush=True)
    if not summary['passed']:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
