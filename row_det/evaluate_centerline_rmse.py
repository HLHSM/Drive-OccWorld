#!/usr/bin/env python3
"""Evaluate finite ego-corridor centerlines against geometric planting rows.

RMSE is conditional on available finite support. Availability and compared
length are reported separately. Label/model main errors use identical frames
and x samples; no distance/heading gate removes large errors.
"""

import argparse
from collections import defaultdict
import csv
import hashlib
import json
from pathlib import Path
import re
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from row_det.generate_row_predictions import source_signature, write_json
from row_det.geometric_evaluation import geometric_rows


def reference_centerline(rows, minimum_length=2.):
    """Select the true ego corridor independently of occupancy/estimates.

    Sort by y-intercept within each field, require adjacent planting row IDs,
    and use finite shared support. The origin must be between the two rows.
    No nearest-to-prediction reassignment or extrapolated error samples.
    """
    fields = defaultdict(list)
    for row in rows:
        points = np.asarray(row['points'], dtype=float)[:, :2]
        if len(points) != 2 or not np.isfinite(points).all():
            raise ValueError('Expected a finite straight geometric GT row')
        points = points[np.argsort(points[:, 0])]
        dx = points[1, 0] - points[0, 0]
        if dx <= 1e-10:
            continue
        slope = (points[1, 1] - points[0, 1]) / dx
        intercept = float(points[0, 1] - slope * points[0, 0])
        fields[row.get('field_id')].append((intercept, str(row['id']), points, float(slope)))
    candidates = []
    for field, members in fields.items():
        members.sort(key=lambda r: (r[0], r[1]))
        for left, right in zip(members, members[1:]):
            bl, il, lp, ml = left
            br, ir, rp, mr = right
            a, b = re.search(r'/row_(\d+)$', il), re.search(r'/row_(\d+)$', ir)
            if a is None or b is None:
                raise ValueError('Geometric row IDs must end in /row_<integer>')
            if abs(int(a[1]) - int(b[1])) != 1:
                continue  # Do not bridge an invalid/missing annotated row.
            if not bl <= 0 <= br:
                continue
            lo, hi = max(0., lp[0, 0], rp[0, 0]), min(lp[1, 0], rp[1, 0])
            if hi - lo < minimum_length:
                continue
            x = np.array([lo, hi])
            if np.any((mr - ml) * x + br - bl <= 0):
                continue
            y = (ml + mr) * x / 2 + (bl + br) / 2
            result = dict(status='geometric_reference', points=np.column_stack((x, y)).tolist(),
                          row_ids=[il, ir], field_id=field, length_x_m=float(hi - lo))
            candidates.append((abs((bl + br) / 2), str(field), il, ir, result))
    return min(candidates, key=lambda c: c[:4])[-1] if candidates else dict(
        status='no_geometric_ego_corridor', points=[], row_ids=[], length_x_m=0.)


def xy_points(line):
    points = np.asarray(line.get('points', []), dtype=float)
    if points.size == 0:
        return None
    if (points.ndim != 2 or points.shape[1] < 2 or len(points) < 2
            or not np.isfinite(points).all() or np.any(np.diff(points[:, 0]) <= 0)):
        raise ValueError('Centerline must be finite, forward-monotone Nx2/Nx3')
    return points[:, :2]


def compare_centerlines(reference, estimates, sample_step=.05, minimum_length=2.):
    """Compare all estimates on the SAME finite x samples, without error gates."""
    if sample_step <= 0 or minimum_length <= 0:
        raise ValueError('Sampling step and minimum length must be positive')
    ref = xy_points(reference)
    if ref is None:
        return dict(status='no_reference', samples=0, compared_length_m=0., scores={})
    points = {side: xy_points(line) for side, line in estimates.items()}
    if not points or any(p is None for p in points.values()):
        return dict(status='missing_estimate', samples=0, compared_length_m=0., scores={})
    lo = max([ref[0, 0]] + [p[0, 0] for p in points.values()])
    hi = min([ref[-1, 0]] + [p[-1, 0] for p in points.values()])
    if hi - lo < minimum_length - 1e-9:
        return dict(status='insufficient_common_support', samples=0, compared_length_m=0., scores={})
    x = np.linspace(lo, hi, max(2, int(np.ceil((hi - lo) / sample_step)) + 1))
    truth = np.interp(x, ref[:, 0], ref[:, 1])
    scores = {}
    for side, p in points.items():
        error = np.interp(x, p[:, 0], p[:, 1]) - truth
        scores[side] = dict(squared_error_sum_m2=float(error @ error),
            absolute_error_sum_m=float(np.abs(error).sum()), signed_error_sum_m=float(error.sum()),
            rmse_m=float(np.sqrt(np.mean(error ** 2))), mae_m=float(np.mean(np.abs(error))),
            max_absolute_error_m=float(np.max(np.abs(error))))
    return dict(status='evaluated', samples=len(x), compared_length_m=float(hi - lo),
                support_x_m=[float(lo), float(hi)], scores=scores)


def pooled_scores(comparisons, side):
    valid = [c for c in comparisons if c['status'] == 'evaluated']
    n = sum(c['samples'] for c in valid)
    sse = sum(c['scores'][side]['squared_error_sum_m2'] for c in valid)
    sae = sum(c['scores'][side]['absolute_error_sum_m'] for c in valid)
    return dict(frames=len(valid), samples=n, squared_error_sum_m2=sse,
        absolute_error_sum_m=sae, compared_length_m=sum(c['compared_length_m'] for c in valid),
        rmse_m=float(np.sqrt(sse / n)) if n else None, mae_m=sae / n if n else None)


def summarize(records):
    paired = [r['paired'] for r in records]
    reference_frames = sum(bool(r['reference']['points']) for r in records)
    result = dict(frames=len(records), reference_frames=reference_frames,
        reference_total_length_m=sum(r['reference']['length_x_m'] for r in records),
        paired_frames=sum(c['status'] == 'evaluated' for c in paired), sides={})
    for side in ('gt', 'pred'):
        generated = sum(r['generated'][side] for r in records)
        own = pooled_scores([r['individual'][side] for r in records], side)
        shared = pooled_scores(paired, side)
        result['sides'][side] = dict(generated_frames=generated,
            generation_rate=generated / len(records) if records else None,
            individually_comparable=own, paired_comparison=shared)
    length = result['sides']['gt']['paired_comparison']['compared_length_m']
    result['paired_reference_length_coverage'] = length / result['reference_total_length_m'] if result['reference_total_length_m'] else None
    return result


def replace_inline_block(source, name, body, legacy_input):
    """Update only the generated block, preserving hand-edited manuscript text."""
    begin, end = f'% BEGIN GENERATED CENTERLINE RMSE {name}', f'% END GENERATED CENTERLINE RMSE {name}'
    block = begin + '\n' + body.rstrip() + '\n' + end
    if begin in source or end in source:
        if source.count(begin) != 1 or source.count(end) != 1:
            raise ValueError(f'Incomplete or duplicate inline {name} markers')
        start, stop = source.index(begin), source.index(end)
        if stop < start:
            raise ValueError(f'Reversed inline {name} markers')
        return source[:start] + block + source[stop + len(end):]
    if source.count(legacy_input) != 1:
        raise ValueError(f'Cannot locate inline {name} block or legacy input')
    return source.replace(legacy_input, block, 1)


def write_paper(summary, paper):
    overall = summary['overall']
    table = [r'\begin{table}[pos=htbp]', r'\centering',
        r'\caption{行间几何中心线误差。没有中心线的帧不参与误差计算；两组误差使用相同的 ' + f"{overall['paired_frames']:,}"
        + r' 帧及相同纵向采样点，单位为 m；生成率按全部 ' + f"{overall['frames']:,}"
        + r' 帧统计。RMSE、MAE 越低越好。}', r'\label{tab:crop_centerline_rmse}',
        r'\begin{tabular*}{\linewidth}{@{\extracolsep{\fill}}lrrr}', r'\toprule',
        r'中心线来源 & 横向 RMSE & 横向 MAE & 生成帧数（生成率）\\', r'\midrule']
    for side, label in [('gt', '语义占用标签'), ('pred', 'AgriOcc 预测占用')]:
        m = overall['sides'][side]
        error = m['paired_comparison']
        format_error = lambda v: '--' if v is None else f'{v:.3f}'
        table.append(label + ' & ' + format_error(error['rmse_m']) + ' & ' + format_error(error['mae_m'])
            + f" & {m['generated_frames']:,}（{100*m['generation_rate']:.1f}\\%）" + r' \\')
    table += [r'\bottomrule', r'\end{tabular*}', r'\end{table}']
    (paper / 'data').mkdir(exist_ok=True)
    gt = overall['sides']['gt']['paired_comparison']
    pred = overall['sides']['pred']['paired_comparison']
    if gt['rmse_m'] is not None:
        paragraph = (f"几何真值可定义参考中心线的帧数为 {overall['reference_frames']:,}，其中 {overall['paired_frames']:,} 帧同时具有标签与预测中心线并满足共同支持条件。"
            f"在这组相同帧、相同采样点上，标签占用与预测占用的横向 RMSE 分别为 {gt['rmse_m']:.3f} m 和 {pred['rmse_m']:.3f} m。"
            f"逐帧累加的共同评价长度为 {gt['compared_length_m']:,.1f} m，占全部真值中心线纵向支持长度的 {100*overall['paired_reference_length_coverage']:.1f}\\%。\n")
    else:
        paragraph = '本次数据中没有满足共同支持条件的帧，因此不报告中心线 RMSE。\n'
    main = paper / 'main.tex'
    source = replace_inline_block(main.read_text(), 'TABLE', '\n'.join(table),
                                 r'\input{figures/crop_centerline_rmse_table.tex}')
    source = replace_inline_block(source, 'RESULTS', paragraph,
                                 r'\input{figures/crop_centerline_rmse_results.tex}')
    main.write_text(source)
    write_json(paper / 'data' / 'crop_centerline_rmse.json', summary)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('evaluation_dir', type=Path)
    parser.add_argument('--row-prediction-dir', type=Path)
    parser.add_argument('--output-dir', type=Path)
    parser.add_argument('--paper-dir', type=Path)
    parser.add_argument('--sample-step', type=float, default=.05)
    parser.add_argument('--min-common-length', type=float, default=2.)
    args = parser.parse_args()
    if args.sample_step <= 0 or args.min_common_length <= 0:
        parser.error('Sampling and support lengths must be positive')
    base = args.evaluation_dir.resolve()
    row_dir = args.row_prediction_dir or base / 'row_prediction_centerline'
    output = args.output_dir or base / 'centerline_rmse'
    (output / 'per_frame').mkdir(parents=True, exist_ok=True)
    paths = sorted((base / 'prediction').glob('*.npz'))
    if not paths:
        raise FileNotFoundError('No saved occupancy predictions')
    run = json.loads((base / 'run_config.json').read_text())
    records, digest = [], hashlib.sha256()
    for index, prediction in enumerate(paths, 1):
        path = row_dir / (prediction.stem + '.json')
        raw = path.read_bytes()
        digest.update(path.name.encode() + b'\0' + raw + b'\0')
        item = json.loads(raw)
        if item['source'] != source_signature(prediction):
            raise ValueError(f'Stale row artifact: {path}')
        frame = item['frame']
        # Recheck geometric GT against the current world annotations and grid pose.
        sequence = Path(run['data_root']) / frame['sequence_path']
        meta = json.loads((sequence / 'meta' / (frame['frame_id'] + '.json')).read_text())
        truth, _ = geometric_rows(meta, sequence, item['point_cloud_range'])
        saved_truth = item['current']['geometry']['rows']
        if len(truth) != len(saved_truth):
            raise ValueError(f'World GT/grid transform changed: {path}')
        for live, saved in zip(truth, saved_truth):
            if live['id'] != saved['id'] or live.get('field_id') != saved.get('field_id'):
                raise ValueError(f'Geometric row identity changed: {path}')
            np.testing.assert_allclose(live['points'], saved['points'], rtol=0, atol=1e-9)
        reference = reference_centerline(truth, args.min_common_length)
        estimates = {side: item['current'][side]['geometric_centerline'] for side in ('gt', 'pred')}
        settings = dict(sample_step=args.sample_step, minimum_length=args.min_common_length)
        record = dict(frame=frame, source_row_file=path.name, reference=reference,
            generated={side: xy_points(line) is not None for side, line in estimates.items()},
            individual={side: compare_centerlines(reference, {side: line}, **settings)
                        for side, line in estimates.items()},
            paired=compare_centerlines(reference, estimates, **settings))
        records.append(record)
        write_json(output / 'per_frame' / path.name, record)
        if index % 200 == 0 or index == len(paths):
            print(f'Evaluated {index}/{len(paths)} frames', flush=True)
    protocol = dict(version='ego_centerline_lateral_rmse_v1', units='metres',
        reference='midline_of_adjacent_valid_geometric_rows_bracketing_origin_within_same_field',
        adjacent_row_ids_required=True, minimum_common_x_length_m=args.min_common_length,
        sample_step_x_m=args.sample_step, error='estimated_y_minus_geometric_y_at_same_x',
        primary_cohort='same_frames_and_x_samples_for_label_model_and_geometry',
        extrapolation=False, distance_or_angle_error_filter=False,
        pooling='sqrt(sum_all_sample_squared_errors / total_sample_count)',
        missing_estimate_error=None, generation_rate_denominator=len(paths),
        limitations=['Conditional localization error, not a complete-map detection or driving score.',
                     'Missing centerlines have no numeric RMSE; report availability separately.',
                     'Individual-side RMSE uses different cohorts; paired RMSE is the primary comparison.'])
    by_crop = {crop: summarize([r for r in records if r['frame']['crop_type'] == crop])
               for crop in sorted({r['frame']['crop_type'] for r in records})}
    summary = dict(evaluation_dir=str(base), row_prediction_dir=str(row_dir.resolve()),
        checkpoint=run['checkpoint'], protocol=protocol, source_artifacts_sha256=digest.hexdigest(),
        evaluator_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        overall=summarize(records), by_crop=by_crop)
    write_json(output / 'metrics.json', summary)
    rows = []
    for r in records:
        row = dict(sample_index=r['frame']['sample_index'], crop_type=r['frame']['crop_type'],
            geometric_reference_available=bool(r['reference']['points']), paired_status=r['paired']['status'],
            samples=r['paired']['samples'], compared_length_m=r['paired']['compared_length_m'])
        for side in ('gt', 'pred'):
            row[side + '_generated'] = r['generated'][side]
            row[side + '_rmse_m'] = r['paired']['scores'].get(side, {}).get('rmse_m')
        rows.append(row)
    with (output / 'per_frame_metrics.csv').open('w') as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    if args.paper_dir:
        write_paper(summary, args.paper_dir)
    print(json.dumps(summary['overall'], indent=2))


if __name__ == '__main__':
    main()
