#!/usr/bin/env python3
"""Paired row/occupancy comparison for legacy, calibrated and adapted inference."""
import argparse
import json
from pathlib import Path
import pickle
import numpy as np


def occupancy(root, filename='occupancy_metrics.pkl'):
    with (root / filename).open('rb') as handle:
        hist = np.asarray(pickle.load(handle)['hist_for_iou_current'], dtype=float)
    union = hist.sum(0) + hist.sum(1) - hist.diagonal()
    iou = np.divide(hist.diagonal(), union, out=np.zeros(6), where=union > 0)
    return dict(crop_iou=float(iou[1]), miou=float(iou.mean()))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('baseline', type=Path)
    parser.add_argument('calibrated', type=Path)
    parser.add_argument('adapted', type=Path)
    parser.add_argument('--adaptation-dir', type=Path, required=True)
    args = parser.parse_args()
    paths = [args.baseline.resolve(), args.calibrated.resolve(), args.adapted.resolve()]
    summaries = [json.loads((p / 'row_metrics.json').read_text()) for p in paths]
    frame_indices = [json.loads((p / 'frame_index.json').read_text()) for p in paths]
    if any(f != frame_indices[0] for f in frame_indices[1:]):
        raise ValueError('evaluation cohorts differ')
    for summary in summaries[1:]:
        for key in ('protocol', 'detector', 'geometric_rows_total'):
            if summary[key] != summaries[0][key]:
                raise ValueError(f'evaluation {key} changed')
        if (summary['aggregated']['overall']['all']['label_fit_vs_geometry'] !=
                summaries[0]['aggregated']['overall']['all']['label_fit_vs_geometry']):
            raise ValueError('semantic-label baseline changed')
    selection = json.loads((args.adaptation_dir / 'selected_checkpoint.json').read_text())
    plan = json.loads((args.adaptation_dir / 'adaptation_plan.json').read_text())
    validation_controls = {mode: occupancy(args.adaptation_dir, f'original_val_{mode}.pkl')
        for mode in ('legacy', 'grid_world') if (args.adaptation_dir / f'original_val_{mode}.pkl').exists()}
    names = ['Original checkpoint / legacy projection', 'Original checkpoint / world-grid projection',
             'Calibration-adapted checkpoint / world-grid projection']
    rows = []
    for name, path, summary in zip(names, paths, summaries):
        rows.append(dict(name=name, directory=str(path), checkpoint=summary['checkpoint'],
            occupancy=occupancy(path), row_metrics=summary['aggregated']['overall']['all']['model_fit_vs_geometry']))
    baseline = rows[0]['row_metrics']['0.20']['f1']
    result = dict(frames=summaries[0]['frames'], sequences=summaries[0]['sequences'],
        geometry_rows=summaries[0]['geometric_rows_total'], comparisons=rows,
        adaptation=plan, selection=selection, original_validation_controls=validation_controls,
        row_f1_at_020_change_pp=(rows[2]['row_metrics']['0.20']['f1'] - baseline) * 100,
        row_f1_at_020_relative_change_percent=(rows[2]['row_metrics']['0.20']['f1'] / baseline - 1) * 100,
        protocol_unchanged=True, semantic_label_baseline_unchanged=True)
    (paths[2] / 'repair_comparison.json').write_text(json.dumps(result, indent=2) + '\n')
    lines = ['# Grid-to-camera calibration repair', '',
        f"Paired evaluation: {result['frames']} frames, {result['sequences']} sequences, "
        f"{result['geometry_rows']} clipped geometric GT rows.", '',
        '| Pipeline | Crop voxel IoU | Voxel mIoU | Row F1@0.10m | Row F1@0.20m | Row F1@0.30m |',
        '| --- | ---: | ---: | ---: | ---: | ---: |']
    for row in rows:
        metrics = row['row_metrics']
        lines.append('| ' + row['name'] + ' | ' +
            ' | '.join(f'{v:.4f}' for v in [row['occupancy']['crop_iou'], row['occupancy']['miou'],
                                           metrics['0.10']['f1'], metrics['0.20']['f1'], metrics['0.30']['f1']]) + ' |')
    lines += ['', '## Diagnosis and implementation', '',
        'The old adapter uses the nominal camera mount transform as if it were a camera pose in the occupancy grid. '
        'The exported grid is ground-anchored and yaw-level, while the real camera follows vehicle pitch/roll and component-root offsets. '
        'This inconsistency also exists in the original training data; the old weights have adapted to the old image sampling.', '',
        '`camera_geometry=grid_world` now computes `optical_from_grid = UE_to_optical @ inverse(world_from_camera) @ world_from_grid`, '
        'using each camera world pose and semantic_occupancy.grid_frame_world_ue. '
        'Temporal motion uses the same grid origin. No row annotations or occupancy classes enter calibration. '
        'Legacy remains the default for checkpoint reproduction. '
        '[UE FRotator convention](https://dev.epicgames.com/documentation/en-us/unreal-engine/API/Runtime/Core/TRotator): '
        'column-vector rotation is Rz(yaw) Ry(-pitch) Rx(-roll).', '',
        '## Adaptation and checkpoint selection', '',
        f"Only the original training split was used: {plan['train_frames']} frames, 8 uniformly selected per sequence. "
        f"Original validation has {plan['val_frames']} frames, 4 per sequence. "
        'The image backbone is frozen; head/BEV parameters adapt for two epochs, seed 0, LR 5e-5, effective batch 12. '
        'Train/validation scenarios are disjoint. SimData-croprow is not used for gradient updates or checkpoint selection.', '',
        '| Adaptation epoch | Original-validation crop IoU | Original-validation mIoU |',
        '| --- | ---: | ---: |']
    for c in selection['candidates']:
        lines.append(f"| {c['epoch']} | {c['val_crop_iou']:.5f} | {c['val_miou']:.5f} |")
    validation_note = ''
    if set(validation_controls) == {'legacy', 'grid_world'}:
        validation_note = (f"On this same original-validation subset, old weights score crop IoU "
            f"{validation_controls['legacy']['crop_iou']:.4f} with legacy projection and "
            f"{validation_controls['grid_world']['crop_iou']:.4f} with world-grid projection. ")
    lines += ['', f"Selected epoch {selection['selected']['epoch']} by original-validation crop IoU. "
        + validation_note + 'Short adaptation does not establish a universal improvement over the old checkpoint.', '',
        '## Row protocol and interpretation', '',
        'The detector, ROI, GT clipping, <=0.05m complete-reference sampling, strict delta threshold, '
        '>=85% coverage and maximum-cardinality one-to-one matching are unchanged. '
        'The label-occupancy baseline is exactly unchanged. Scores are pooled TP/FP/FN micro F1. '
        'No overlap-only comparison, line extension, GT direction, GT mask or threshold relaxation is used.', '',
        f"At 0.20m the adapted-minus-original row F1 difference is {result['row_f1_at_020_change_pp']:+.2f} percentage points "
        f"({result['row_f1_at_020_relative_change_percent']:+.2f}% relative). "
        'The table reports voxel and row metrics together so that a gain in row fitting is not mistaken for an across-the-board model gain.', '',
        'This is a repair/development comparison on already inspected evaluation data, not a new blind test. '
        'Two adaptation epochs with a frozen image backbone are a limited calibration experiment. '
        'Residual domain shift, occupancy boundary errors and row-fit errors remain; '
        'the 0.2m output grid and complete-row coverage criterion make small angular/lateral errors consequential.', '',
        '## Artifacts', '',
        f"- Selected checkpoint: `{selection['selected']['path']}`.",
        '- `prediction/`, `row_prediction/`, `row_metrics.json`: complete new predictions and metrics.',
        '- `visualization/`: 100 three-panel images with exactly the original balanced sequence/window selection and naming.',
        '- `repair_comparison.json`: machine-readable paired results and model selection provenance.',
        '- `REPORT.md`: detailed row evaluation protocol and TP/FP/FN counts.', '']
    (paths[2] / 'REPAIR_REPORT.md').write_text('\n'.join(lines))
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
