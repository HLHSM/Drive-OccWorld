#!/usr/bin/env python3
"""Generate task-local row_prediction/*.json from saved occupancy NPZs."""

import argparse
from collections import Counter
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import time

import numpy as np

# Also support `python /path/to/row_det/generate_row_predictions.py ...`.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from row_det.detector import ALGORITHM_VERSION, RowConfig, config_dict, detect_rows


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('task_dir', type=Path,
                        help='training task directory, or its occupancy prediction directory')
    parser.add_argument('--prediction-dir', type=Path,
                        help='override task_dir/prediction input directory')
    parser.add_argument('--output-dir', type=Path,
                        help='override task_dir/row_prediction output directory')
    parser.add_argument('--count', type=int, default=-1,
                        help='process first N sorted files; -1 processes all')
    parser.add_argument('--overwrite', action='store_true', help='recompute matching cached artifacts')
    parser.add_argument('--preview-count', type=int, default=0,
                        help='also write N BEV comparison PNGs in row_prediction/previews')
    cfg = RowConfig()
    options = {
        'crop_id': int, 'z_min': float, 'z_max': float,
        'max_angle_deg': float, 'angle_step_deg': float,
        'heading_prior_strength': float,
        'min_row_spacing': float, 'row_band': float,
        'min_crop_cells': int, 'min_row_points': int,
        'min_row_length': float, 'slice_length': float,
        'residual_threshold': float, 'curve_degree': int,
        'sample_step': float, 'min_path_length': float,
        'vehicle_width': float, 'vehicle_height': float, 'clearance_margin': float, 'seed': int,
    }
    for name, value_type in options.items():
        parser.add_argument('--' + name.replace('_', '-'), type=value_type,
                            default=getattr(cfg, name),
                            help=f'{name} (default: {getattr(cfg, name)})')
    args = parser.parse_args()
    if args.count == 0 or args.count < -1 or args.preview_count < 0:
        parser.error('--count must be -1 or positive; --preview-count must be nonnegative')
    return args, RowConfig(**{name: getattr(args, name) for name in options})


def source_signature(path):
    stat = path.stat()
    return dict(name=path.name, size_bytes=stat.st_size, mtime_ns=stat.st_mtime_ns)


def compare_rows(pred, gt, distance_threshold=.4):
    """One-to-one row matches over shared x support; a diagnostic, not GT lines.

    These 'GT rows' are derived by this same geometric extractor from semantic
    GT, so the measurements do not substitute for labelled planting lines.
    """
    predicted, truth = pred['rows'], gt['rows']
    metrics = dict(pred_rows=len(predicted), gt_rows=len(truth), matched_rows=0,
                   mean_lateral_error_m=None, mean_heading_error_deg=None,
                   match_distance_threshold_m=distance_threshold)
    if not predicted or not truth:
        return metrics
    costs = np.full((len(predicted), len(truth)), 1e6)
    headings = np.zeros_like(costs)
    for i, row in enumerate(predicted):
        p = np.asarray(row['points'])
        for j, target in enumerate(truth):
            g = np.asarray(target['points'])
            start, end = max(p[0, 0], g[0, 0]), min(p[-1, 0], g[-1, 0])
            if end - start < 1.0:
                continue
            x = np.linspace(start, end, 30)
            py, gy = np.interp(x, p[:, 0], p[:, 1]), np.interp(x, g[:, 0], g[:, 1])
            heading = np.rad2deg(np.abs(np.arctan(np.gradient(py, x)) -
                                         np.arctan(np.gradient(gy, x)))).mean()
            if heading <= 10.0:
                costs[i, j] = np.abs(py - gy).mean()
                headings[i, j] = heading
    # Greedy nearest matching with gates. Each extracted GT/pred row is used
    # at most once; this lightweight diagnostic does not require SciPy.
    options = sorted((costs[i, j], i, j) for i in range(len(predicted))
                     for j in range(len(truth)) if costs[i, j] <= distance_threshold)
    pairs, used_pred, used_gt = [], set(), set()
    for _, i, j in options:
        if i not in used_pred and j not in used_gt:
            pairs.append((i, j))
            used_pred.add(i)
            used_gt.add(j)
    if pairs:
        metrics.update(matched_rows=len(pairs),
                       mean_lateral_error_m=round(float(np.mean([costs[i, j] for i, j in pairs])), 4),
                       mean_heading_error_deg=round(float(np.mean([headings[i, j] for i, j in pairs])), 4))
    return metrics


def build_artifact(path, cfg):
    with np.load(path, allow_pickle=False) as data:
        if 'num_classes' in data and int(data['num_classes']) != 6:
            raise ValueError('row detector expects the six-class FarmSim taxonomy')
        pc_range = data['point_cloud_range'].astype(float)
        pred = data['current_pred']
        artifact = dict(schema_version=1, algorithm_version=ALGORITHM_VERSION,
                        source=source_signature(path),
                        coordinate_frame='ego_x_forward_y_right_z_up_metres',
                        point_cloud_range=pc_range.tolist(), occupancy_shape=list(pred.shape),
                        config=config_dict(cfg), current={}, future=[])
        artifact['current']['pred'] = detect_rows(pred, pc_range, cfg)
        if 'current_gt' in data:
            if data['current_gt'].shape != pred.shape:
                raise ValueError('current_pred/current_gt shapes differ')
            artifact['current']['gt'] = detect_rows(data['current_gt'], pc_range, cfg)
            artifact['current']['comparison'] = compare_rows(
                artifact['current']['pred'], artifact['current']['gt'])
        if 'future_pred' in data:
            for t, frame in enumerate(data['future_pred']):
                result = dict(pred=detect_rows(frame, pc_range, cfg))
                if 'future_gt' in data and t < len(data['future_gt']):
                    result['gt'] = detect_rows(data['future_gt'][t], pc_range, cfg)
                    result['comparison'] = compare_rows(result['pred'], result['gt'])
                artifact['future'].append(result)
    return artifact


def write_json(path, payload):
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def render_preview(source, artifact, output):
    import matplotlib
    matplotlib.use('Agg')
    from matplotlib import pyplot as plt
    with np.load(source, allow_pickle=False) as data:
        panels = [('gt', 'Semantic GT'), ('pred', 'Occupancy prediction')]
        fig, axes = plt.subplots(1, 2, figsize=(12, 6), constrained_layout=True)
        pc = artifact['point_cloud_range']
        for ax, (side, title) in zip(axes, panels):
            key = 'current_' + side
            if key not in data or side not in artifact['current']:
                ax.set_visible(False)
                continue
            density = (data[key] == artifact['config']['crop_id']).sum(axis=2)
            ax.imshow(density.T, origin='lower', extent=[pc[0], pc[3], pc[1], pc[4]],
                      cmap='Greens', interpolation='nearest', aspect='equal')
            result = artifact['current'][side]
            for row in result['rows']:
                points = np.asarray(row['points'])
                ax.plot(points[:, 0], points[:, 1], color='#db2589', lw=1.8)
                ax.text(*points[len(points) // 2, :2], str(row['id']), color='#b5005c')
            reference = result['reference_path']
            if reference['points']:
                points = np.asarray(reference['points'])
                ax.plot(points[:, 0], points[:, 1], color='#00a8e8', lw=2.5, label='reference')
            ax.plot(0, 0, 'r>', ms=9)
            ax.set(xlabel='forward x (m)', ylabel='right y (m)', xlim=(pc[0], pc[3]),
                   ylim=(pc[1], pc[4]), title=f"{title}: {len(result['rows'])} rows\n{reference['status']}")
        fig.suptitle(source.stem, fontsize=8)
        fig.savefig(output, dpi=150)
        plt.close(fig)


def main():
    args, cfg = parse_args()
    cfg.validate()
    task = args.task_dir.resolve()
    is_prediction_dir = any(task.glob('*.npz'))
    prediction_dir = (args.prediction_dir or (task if is_prediction_dir else task / 'prediction')).resolve()
    task_root = task.parent if is_prediction_dir else task
    output_dir = (args.output_dir or task_root / 'row_prediction').resolve()
    paths = sorted(prediction_dir.glob('*.npz'))
    if args.count > 0:
        paths = paths[:args.count]
    if not paths:
        raise FileNotFoundError(f'No occupancy NPZ files in {prediction_dir}')
    output_dir.mkdir(parents=True, exist_ok=True)
    start = time.monotonic()
    summaries, failures = [], []
    for index, path in enumerate(paths, 1):
        destination = output_dir / (path.stem + '.json')
        try:
            artifact = None
            if destination.exists() and not args.overwrite:
                try:
                    cached = json.loads(destination.read_text())
                    if (cached.get('schema_version') == 1 and cached.get('algorithm_version') == ALGORITHM_VERSION
                            and cached.get('source') == source_signature(path)
                            and cached.get('config') == config_dict(cfg)):
                        artifact = cached
                except (ValueError, OSError):
                    pass
            if artifact is None:
                artifact = build_artifact(path, cfg)
                write_json(destination, artifact)
            if index <= args.preview_count:
                preview_dir = output_dir / 'previews'
                preview_dir.mkdir(exist_ok=True)
                render_preview(path, artifact, preview_dir / (path.stem + '.png'))
            frame = artifact['current']
            summaries.append(dict(name=path.stem,
                pred_rows=len(frame['pred']['rows']),
                pred_reference_status=frame['pred']['reference_path']['status'],
                gt_rows=len(frame.get('gt', {}).get('rows', [])),
                gt_reference_status=frame.get('gt', {}).get('reference_path', {}).get('status'),
                comparison=frame.get('comparison')))
        except (ValueError, OSError, KeyError) as error:
            failures.append(dict(name=path.name, error=str(error)))
            print(f'Failed: {path.name}: {error}', file=sys.stderr, flush=True)
        if index == len(paths) or index % 10 == 0:
            print(f'Processed {index}/{len(paths)} occupancy files.', flush=True)
    summary = dict(schema_version=1, created_at=datetime.now(timezone.utc).isoformat(),
                   prediction_dir=str(prediction_dir), output_dir=str(output_dir),
                   config=config_dict(cfg), processed=len(summaries), failures=failures,
                   elapsed_seconds=round(time.monotonic() - start, 3),
                   pred_reference_status_counts=dict(Counter(s['pred_reference_status'] for s in summaries)),
                   gt_reference_status_counts=dict(Counter(s['gt_reference_status'] for s in summaries)),
                   samples=summaries)
    write_json(output_dir / 'summary.json', summary)
    print(f"Saved {len(summaries)} row artifacts to {output_dir}")
    print('Prediction references:', summary['pred_reference_status_counts'])
    if failures:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
