#!/usr/bin/env python3
"""Compare geometry GT, label-occupancy fits and model-occupancy fits."""

import argparse
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import sys
import time

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from row_det.detector import ALGORITHM_VERSION, RowConfig, config_dict, detect_rows
from row_det.generate_row_predictions import source_signature, write_json
from row_det.geometric_evaluation import counts_metrics, evaluate_rows, geometric_rows


COMPARISONS = ('label_fit_vs_geometry', 'model_fit_vs_geometry', 'model_fit_vs_label_fit')
PROTOCOL_VERSION = 'grid_xy_full_reference_coverage_v1'


def expected_stem(frame):
    scene = re.sub(r'[^A-Za-z0-9_.-]+', '_', frame['scenario_id'])
    return f"{frame['sample_index']:06d}_{scene}_{frame['frame_id']}"


def process_frame(frame, prediction_path, data_root, output_path, cfg_dict, protocol):
    cfg = RowConfig(**cfg_dict)
    source = source_signature(prediction_path)
    sequence = Path(data_root) / frame['sequence_path']
    metadata_path = sequence / 'meta' / (frame['frame_id'] + '.json')
    meta = json.loads(metadata_path.read_text())
    annotation_source = dict(data_root=str(Path(data_root).resolve()),
        metadata=source_signature(metadata_path),
        layout=source_signature(sequence / meta['evaluation_ground_truth']['crop_rows_file']))
    if output_path.exists():
        cached = json.loads(output_path.read_text())
        if (cached.get('source') == source and cached.get('config') == cfg_dict
                and cached.get('evaluation_protocol') == protocol
                and cached.get('annotation_source') == annotation_source
                and cached.get('frame') == frame
                and cached.get('algorithm_version') == ALGORITHM_VERSION):
            return frame['sample_index'], cached['current']['metrics'], cached['geometry_diagnostics']
    with np.load(prediction_path, allow_pickle=False) as data:
        if int(data['sample_index']) != frame['sample_index']:
            raise ValueError('saved sample index does not match manifest')
        if int(data['num_classes']) != 6:
            raise ValueError('expected six-class FarmSim occupancy')
        pc = data['point_cloud_range'].astype(float)
        prediction, label = data['current_pred'], data['current_gt']
        if prediction.shape != label.shape:
            raise ValueError('label and predicted occupancy shapes differ')
        truth, geometry_diagnostics = geometric_rows(meta, sequence, pc)
        if len(truth) != frame['geometry_row_count']:
            raise ValueError('geometry annotations changed after dataset preparation')
        # In particular, no GT mask or geometric row orientation is fed into
        # the model's row fitter. Both sides run the identical fixed detector.
        pred_rows = detect_rows(prediction, pc, cfg)
        label_rows = detect_rows(label, pc, cfg)
        bounds = pc[[0, 1, 3, 4]]
        settings = dict(thresholds=protocol['thresholds_m'],
                        sample_step=protocol['sample_step_m'],
                        required_coverage=protocol['required_coverage'])
        metrics = dict(
            label_fit_vs_geometry=evaluate_rows(truth, label_rows['rows'], bounds, **settings),
            model_fit_vs_geometry=evaluate_rows(truth, pred_rows['rows'], bounds, **settings),
            model_fit_vs_label_fit=evaluate_rows(label_rows['rows'], pred_rows['rows'], bounds, **settings))
        artifact = dict(schema_version=1, algorithm_version=ALGORITHM_VERSION,
                        evaluation_protocol=protocol, source=source, frame=frame,
                        annotation_source=annotation_source,
                        coordinate_frame='occupancy_grid_x_forward_y_right_z_up_metres',
                        point_cloud_range=pc.tolist(), occupancy_shape=list(label.shape),
                        config=cfg_dict, geometry_diagnostics=geometry_diagnostics,
                        current=dict(geometry=dict(rows=truth), gt=label_rows,
                                     pred=pred_rows, metrics=metrics), future=[])
    write_json(output_path, artifact)
    return frame['sample_index'], metrics, geometry_diagnostics


def aggregate(frames, results, thresholds):
    counts = defaultdict(lambda: np.zeros(3, dtype=np.int64))
    for frame in frames:
        metrics = results[frame['sample_index']][0]
        groups = [('overall', 'all'), ('crop_type', frame['crop_type']),
                  ('growth_stage', frame['growth_stage']), ('sequence', frame['sequence_path'])]
        for group_type, group in groups:
            for comparison in COMPARISONS:
                for threshold in thresholds:
                    key = f'{threshold:.2f}'
                    item = metrics[comparison][key]
                    counts[(group_type, group, comparison, key)] += [item['tp'], item['fp'], item['fn']]
    output = {}
    for (group_type, group, comparison, threshold), count in counts.items():
        output.setdefault(group_type, {}).setdefault(group, {}).setdefault(comparison, {})[threshold] = counts_metrics(*count)
    return output


def render_comparison(prediction_path, artifact, output_path, color_threshold=.2):
    import matplotlib
    matplotlib.use('Agg')
    from matplotlib import pyplot as plt
    from matplotlib.lines import Line2D
    with np.load(prediction_path, allow_pickle=False) as data:
        label_density = (data['current_gt'] == 1).sum(axis=2)
        pred_density = (data['current_pred'] == 1).sum(axis=2)
    pc = artifact['point_cloud_range']
    current = artifact['current']
    truth = current['geometry']['rows']
    fig, axes = plt.subplots(1, 3, figsize=(16.5, 6.4))
    titles = ['Geometric ground truth', 'Semantic-label occupancy fit', 'Model-predicted occupancy fit']
    for column, (ax, title) in enumerate(zip(axes, titles)):
        density = label_density if column < 2 else pred_density
        if column:
            ax.imshow(np.sqrt(density.T), origin='lower', extent=[pc[0], pc[3], pc[1], pc[4]],
                      cmap='Greens', alpha=.5, interpolation='nearest', aspect='equal')
        for row in truth:
            points = np.asarray(row['points'])
            ax.plot(points[:, 0], points[:, 1], color='#215bbb' if column == 0 else '#333333',
                    lw=1.8 if column == 0 else 1.1, ls='-' if column == 0 else '--',
                    alpha=1 if column == 0 else .7, zorder=2)
            if column == 0:
                ax.plot(points[:, 0], points[:, 1], '.', color='#215bbb', ms=3)
        footer = f'{len(truth)} finite GT rows; full clipped extent'
        if column:
            side = 'gt' if column == 1 else 'pred'
            comparison = 'label_fit_vs_geometry' if column == 1 else 'model_fit_vs_geometry'
            metrics = current['metrics'][comparison]
            matched = {match['estimate_index'] for match in metrics[f'{color_threshold:.2f}']['matches']}
            for index, row in enumerate(current[side]['rows']):
                points = np.asarray(row['points'])
                ax.plot(points[:, 0], points[:, 1], color='#bd11a7' if index in matched else '#e15116',
                        lw=2, zorder=3)
            footer = '\n'.join(
                f"delta={key} m: F1={'N/A' if item['f1'] is None else format(item['f1'], '.3f')}  "
                f"TP/FP/FN={item['tp']}/{item['fp']}/{item['fn']}"
                for key, item in metrics.items())
            title += f" ({len(current[side]['rows'])} rows)"
        else:
            title += f' ({len(truth)} rows)'
        ax.set(title=title, xlim=(pc[0], pc[3]), ylim=(pc[1], pc[4]),
               xlabel='forward x (m)', ylabel='right y (m)')
        ax.set_aspect('equal')
        ax.grid(alpha=.2)
        ax.text(.5, -.17, footer, transform=ax.transAxes, ha='center', va='top', fontsize=9)
    legend = [Line2D([0], [0], color='#215bbb', label='geometric GT'),
              Line2D([0], [0], color='#333333', ls='--', label='geometric GT overlay'),
              Line2D([0], [0], color='#bd11a7', label=f'matched fit at delta={color_threshold:.2f} m'),
              Line2D([0], [0], color='#e15116', label='unmatched fit')]
    fig.legend(handles=legend, loc='lower center', ncol=4, frameon=False, fontsize=9)
    fig.suptitle(prediction_path.stem + '\n85% full-GT coverage; finite-segment metric; colours at 0.20 m only', fontsize=10)
    fig.subplots_adjust(left=.045, right=.99, top=.85, bottom=.24, wspace=.25)
    fig.savefig(output_path, dpi=140)
    plt.close(fig)


def write_report(root, summary):
    overall = summary['aggregated']['overall']['all']
    lines = ['# Crop-row evaluation', '',
             f"Checkpoint: `{summary['checkpoint']['path']}`.", '',
             f"Dataset: {summary['frames']} frames in {summary['sequences']} sequences; "
             f"{summary['geometric_rows_total']} finite geometric rows after XY clipping.", '',
             '| Estimate (reference = geometric GT) | F1@0.10 m | F1@0.20 m | F1@0.30 m |',
             '| --- | ---: | ---: | ---: |',
             '| Geometric GT identity (oracle) | 1.0000 | 1.0000 | 1.0000 |']
    for label, key in [('Semantic-label occupancy fit', 'label_fit_vs_geometry'),
                       ('Model-predicted occupancy fit', 'model_fit_vs_geometry')]:
        values = [f"{overall[key][t]['f1']:.4f}" if overall[key][t]['f1'] is not None else 'N/A'
                  for t in overall[key]]
        lines.append('| ' + label + ' | ' + ' | '.join(values) + ' |')
    lines += ['', '## Counts and precision/recall', '',
              '| Comparison | delta (m) | TP | FP | FN | Precision | Recall | F1 |',
              '| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |']
    for comparison in COMPARISONS:
        for threshold, item in overall[comparison].items():
            def number(value):
                return 'N/A' if value is None else f'{value:.4f}'
            lines.append(f"| {comparison} | {threshold} | {item['tp']} | {item['fp']} | {item['fn']} | "
                         f"{number(item['precision'])} | {number(item['recall'])} | {number(item['f1'])} |")
    label = overall['label_fit_vs_geometry']['0.20']
    model = overall['model_fit_vs_geometry']['0.20']
    if label['f1'] is not None and model['f1'] is not None:
        relative = ((model['f1'] / label['f1'] - 1) * 100) if label['f1'] else None
        lines += ['', '## Findings', '',
                  f"1. At 0.20 m, label-occupancy fits score {label['f1']:.4f}, "
                  f"while model-occupancy fits score {model['f1']:.4f}. "
                  f"The model-minus-label gap is {(model['f1'] - label['f1']) * 100:.2f} percentage points" +
                  (f" ({relative:.2f}% relative)." if relative is not None else '.'),
                  '2. The label-occupancy comparison measures voxelization plus detector errors. '
                  'The model comparison measures the full occupancy-to-row pipeline; this experiment '
                  'does not isolate neural prediction errors from detector sensitivity.',
                  '3. Use a separate validation set and operational error tolerance to fix the primary threshold. '
                  'Do not tune the detector or select the threshold on these evaluation scores.']
    geometry = summary.get('inference', {}).get('camera_geometry', 'legacy')
    calibration_note = (
        '- Inference uses the saved legacy relative-camera adapter. Grid and vehicle frames differ in '
        'height/tilt; this mode does not correct grid-to-camera calibration.' if geometry == 'legacy' else
        f'- Inference camera_geometry={geometry}: camera world pose and occupancy-grid world pose '
        'define optical-from-grid = axis_conversion @ inverse(world_from_camera) @ world_from_grid. '
        'UE FRotator pitch/roll signs are respected. No crop-row annotations enter calibration.')
    lines += ['', '## Protocol', '',
              '- World-row XY is translated by grid location_cm/100 and rotated by inverse grid yaw. '
              'The per-frame evaluation_ground_truth.grid_frame_world_ue is used, never vehicle_pose_ue.',
              '- fit_valid=false rows and zero-length/outside segments are excluded. '
              'No positive minimum GT length, visibility mask, row-angle gate, or overlap-only gate is imposed.',
              '- All groups use the model ROI x=[0,15], y=[-7.5,7.5] m. '
              'The complete clipped reference segment is sampled including both endpoints, with spacing <=0.05 m.',
              '- A pair is eligible when >=85% of samples have Euclidean XY distance strictly below delta '
              'to the estimated finite polyline. No infinite-line extension or endpoint extrapolation is used.',
              '- Maximum-cardinality one-to-one matching yields TP; every extra estimate is FP, every unmatched reference FN. '
              'Counts are pooled over all frames before computing micro F1=2TP/(2TP+FP+FN). '
              'All-empty per-frame scores are null; frames with no GT still contribute any FP.',
              '- The label and model fits use the same frozen projection-RANSAC detector configuration. '
              'Model fitting uses predicted occupancy only, without GT masks, geometric row direction, or row identities.',
              '- The model_fit_vs_label_fit comparison is a secondary reconstruction diagnostic; '
              'it is not a substitute for geometric-GT evaluation.',
              '- Thresholds 0.10/0.20/0.30 m are all reported. No primary threshold is selected using this test set. '
              'Plot colours use 0.20 m only for display.',
              '- Metric adapted from RowDetr/TuSimple coverage matching to metric grid XY and finite-segment distance: '
              '[RowDetr, section 3.1](https://arxiv.org/html/2412.10525v2#S3.SS1). '
              'These metric scores are not directly comparable to its image-space results.',
              calibration_note,
              '- The occupancy path calls pts_bbox_head(..., only_bev=True) -> transformer.get_bev_features, '
              'which bypasses the object-query reference_points layer. Checkpoint loader diagnostics '
              'are retained in inference.log.',
              '', '## Artifacts', '',
              '- `prediction/`: all frame occupancy NPZs; `row_prediction/`: geometric GT, both fits, matching details, coordinates and diagnostics.',
              '- `row_metrics.json`, `per_frame_metrics.csv`, `grouped_metrics.csv`: pooled and per-frame/group scores.',
              f"- `visualization/`: {summary.get('visualization_count', 100)} sequence-balanced centred-window three-panel comparison PNGs, "
              'using the same stable `<sample>_<scenario>_<frame>_prediction.png` naming as occupancy previews.',
              '- `dataset_audit.json`, `evaluation_manifest.json`, `frame_index.json`, `visualization_indices.txt`: dataset and selection provenance.',
              '- `run_config.json`, `inference.log`, `occupancy_metrics.pkl`: inference/config/checkpoint provenance and occupancy metrics.', '']
    (root / 'REPORT.md').write_text('\n'.join(lines))


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('evaluation_dir', type=Path)
    parser.add_argument('--data-root', type=Path, required=True)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--row-config', type=Path, help='fixed RowConfig JSON override')
    parser.add_argument('--inference-config', type=Path, help='actual config used to generate these NPZs')
    parser.add_argument('--camera-geometry', choices=['legacy', 'grid_world'], default='legacy')
    parser.add_argument('--inference-gpu', type=int, default=0)
    parser.add_argument('--skip-visualization', action='store_true', help='metrics-only diagnostic control run')
    parser.add_argument('--sample-step', type=float, default=.05)
    parser.add_argument('--required-coverage', type=float, default=.85)
    parser.add_argument('--thresholds', type=float, nargs='+', default=[.1, .2, .3])
    args = parser.parse_args()
    if args.workers < 1 or args.sample_step <= 0 or not 0 < args.required_coverage <= 1:
        parser.error('invalid worker count or sampling settings')
    if args.thresholds != [.1, .2, .3] or args.sample_step != .05 or args.required_coverage != .85:
        parser.error('this report implements the fixed protocol: delta=0.1/0.2/0.3, step=0.05, coverage=0.85')
    root = args.evaluation_dir.resolve()
    inference_config = args.inference_config or (args.checkpoint.parent / 'farmsim_occ_front3_s.py')
    if not inference_config.is_file():
        parser.error('supply --inference-config pointing to the config used for inference')
    inference = dict(config=str(inference_config.resolve()), camera_geometry=args.camera_geometry,
                     gpu=args.inference_gpu, image_size=[512, 288], queue_length=0)
    frames = json.loads((root / 'frame_index.json').read_text())
    audit = json.loads((root / 'dataset_audit.json').read_text())
    selected = ([] if args.skip_visualization else
                [int(line) for line in (root / 'visualization_indices.txt').read_text().splitlines() if line.strip()])
    cfg = RowConfig(**(json.loads(args.row_config.read_text()) if args.row_config else {}))
    cfg.validate()
    protocol = dict(version=PROTOCOL_VERSION, thresholds_m=args.thresholds,
                    sample_step_m=args.sample_step, required_coverage=args.required_coverage,
                    distance='euclidean_xy_to_finite_polyline', matching='maximum_cardinality_one_to_one',
                    averaging='micro_pooled_counts', reference_extent='full_roi_clipped_reference_including_endpoints')
    row_dir, viz_dir = root / 'row_prediction', root / 'visualization'
    row_dir.mkdir(exist_ok=True)
    viz_dir.mkdir(exist_ok=True)
    missing = [frame['sample_index'] for frame in frames if not (root / 'prediction' / (expected_stem(frame) + '.npz')).exists()]
    if missing:
        raise FileNotFoundError(f'Missing {len(missing)} occupancy predictions; first indices: {missing[:10]}')
    write_json(root / 'run_config.json', dict(
        created_at=datetime.now(timezone.utc).isoformat(), command=sys.argv,
        checkpoint=dict(path=str(args.checkpoint.resolve()), sha256=sha256(args.checkpoint)),
        data_root=str(args.data_root.resolve()), protocol=protocol, detector=config_dict(cfg),
        detector_algorithm=ALGORITHM_VERSION,
        inference_config=str(inference_config.resolve()),
        inference_overrides={'data.test.ann_file': str(root / 'evaluation_manifest.json'),
            'data.test.data_root': str(args.data_root.resolve()), 'data.test.samples_per_gpu': 2,
            'data.workers_per_gpu': 2, 'data.test.camera_geometry': args.camera_geometry},
        inference_gpu=args.inference_gpu, camera_geometry=args.camera_geometry,
        image_size=[512, 288], queue_length=0,
        prediction_mask='none', geometric_gt_mask='XY_ROI_only'))
    start = time.monotonic()
    results, errors = {}, []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        jobs = {pool.submit(process_frame, frame, root / 'prediction' / (expected_stem(frame) + '.npz'),
                            args.data_root, row_dir / (expected_stem(frame) + '.json'), config_dict(cfg), protocol): frame
                for frame in frames}
        for count, job in enumerate(as_completed(jobs), 1):
            frame = jobs[job]
            try:
                index, metrics, diagnostics = job.result()
                results[index] = (metrics, diagnostics)
            except Exception as error:
                errors.append(dict(sample_index=frame['sample_index'], error=repr(error)))
                print('FAILED', errors[-1], flush=True)
            if count % 100 == 0 or count == len(frames):
                print(f'Rows evaluated: {count}/{len(frames)}; elapsed={time.monotonic() - start:.1f}s', flush=True)
    if errors:
        write_json(root / 'failures.json', errors)
        raise RuntimeError(f'{len(errors)} frames failed; no partial aggregate is reported')
    aggregated = aggregate(frames, results, args.thresholds)
    summary = dict(schema_version=1, frames=len(frames), sequences=audit['sequences'],
                   geometric_rows_total=audit['geometric_rows_total'],
                   frames_without_geometric_rows=audit['frames_without_geometric_rows'],
                   protocol=protocol, detector=config_dict(cfg), aggregated=aggregated,
                   inference=inference,
                   visualization_count=len(selected),
                   checkpoint=dict(path=str(args.checkpoint.resolve()), sha256=sha256(args.checkpoint)),
                   extraction_seconds=round(time.monotonic() - start, 3))
    write_json(root / 'row_metrics.json', summary)
    fields = ['sample_index', 'sequence_path', 'frame_id', 'crop_type', 'growth_stage', 'comparison', 'delta_m',
              'tp', 'fp', 'fn', 'precision', 'recall', 'f1']
    with (root / 'per_frame_metrics.csv').open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for frame in frames:
            for comparison, thresholds in results[frame['sample_index']][0].items():
                for threshold, metrics in thresholds.items():
                    writer.writerow(dict({key: frame[key] for key in fields[:5]}, comparison=comparison,
                                         delta_m=threshold, **{key: metrics[key] for key in fields[7:]}))
    with (root / 'grouped_metrics.csv').open('w', newline='') as handle:
        writer = csv.writer(handle)
        writer.writerow(['group_type', 'group', 'comparison', 'delta_m', 'tp', 'fp', 'fn', 'precision', 'recall', 'f1'])
        for group_type, groups in aggregated.items():
            for group, comparisons in groups.items():
                for comparison, thresholds in comparisons.items():
                    for threshold, metrics in thresholds.items():
                        writer.writerow([group_type, group, comparison, threshold] + [metrics[k] for k in ['tp', 'fp', 'fn', 'precision', 'recall', 'f1']])
    write_report(root, summary)
    for count, index in enumerate(selected, 1):
        frame = frames[index]
        stem = expected_stem(frame)
        artifact = json.loads((row_dir / (stem + '.json')).read_text())
        render_comparison(root / 'prediction' / (stem + '.npz'), artifact,
                          viz_dir / (stem + '_prediction.png'))
        if count % 10 == 0 or count == len(selected):
            print(f'Visualizations: {count}/{len(selected)}', flush=True)
    print(json.dumps(aggregated['overall']['all'], indent=2), flush=True)
    print(f'Complete: {root}', flush=True)


if __name__ == '__main__':
    main()
