#!/usr/bin/env python3
"""Aggregate crop IoU/precision/recall in forward-distance bins from NPZs."""

import argparse
import json
from pathlib import Path

import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('prediction_dir')
    parser.add_argument('--edges', type=float, nargs='+',
                        default=[0, 5, 10, 15, 20])
    parser.add_argument('--expected-count', type=int)
    parser.add_argument('--output')
    args = parser.parse_args()
    if len(args.edges) < 2 or any(
            right <= left for left, right in zip(args.edges, args.edges[1:])):
        raise ValueError('--edges must be strictly increasing')
    files = sorted(Path(args.prediction_dir).glob('*.npz'))
    if not files:
        raise FileNotFoundError(f'No NPZ predictions in {args.prediction_dir}')
    if args.expected_count is not None and len(files) != args.expected_count:
        raise ValueError(f'Expected {args.expected_count} predictions, found {len(files)}')

    counts = [dict(tp=0, fp=0, fn=0, gt_crop=0, valid=0,
                   fp_on_soil=0, fp_on_free=0) for _ in args.edges[:-1]]
    point_cloud_range = None
    for path in files:
        with np.load(path) as item:
            pred = item['current_pred']
            gt = item['current_gt']
            pc_range = item['point_cloud_range']
        if pred.shape != gt.shape or pred.ndim != 3:
            raise ValueError(f'{path}: prediction and GT must share [X,Y,Z]')
        if point_cloud_range is None:
            point_cloud_range = pc_range
        elif not np.array_equal(point_cloud_range, pc_range):
            raise ValueError(f'{path}: inconsistent point_cloud_range')
        x_centers = pc_range[0] + (np.arange(gt.shape[0]) + 0.5) * (
            (pc_range[3] - pc_range[0]) / gt.shape[0])
        for (left, right), stat in zip(zip(args.edges, args.edges[1:]), counts):
            x = (x_centers >= left) & (x_centers < right)
            label = gt[x]
            prediction = pred[x]
            valid = label != 255
            crop_gt = (label == 1) & valid
            crop_pred = (prediction == 1) & valid
            stat['tp'] += int(np.count_nonzero(crop_gt & crop_pred))
            stat['fp'] += int(np.count_nonzero(~crop_gt & crop_pred))
            stat['fn'] += int(np.count_nonzero(crop_gt & ~crop_pred))
            stat['gt_crop'] += int(np.count_nonzero(crop_gt))
            stat['valid'] += int(np.count_nonzero(valid))
            stat['fp_on_soil'] += int(np.count_nonzero(crop_pred & (label == 2)))
            stat['fp_on_free'] += int(np.count_nonzero(crop_pred & (label == 0)))

    result = dict(predictions=len(files), crop_class=1,
                  ignore_label=255, range_m=list(map(float, args.edges)), bins=[])
    for (left, right), stat in zip(zip(args.edges, args.edges[1:]), counts):
        tp, fp, fn = stat['tp'], stat['fp'], stat['fn']
        stat.update(range_m=[left, right],
                    iou=tp / (tp + fp + fn) if tp + fp + fn else None,
                    precision=tp / (tp + fp) if tp + fp else None,
                    recall=tp / (tp + fn) if tp + fn else None,
                    gt_crop_fraction=stat['gt_crop'] / stat['valid']
                    if stat['valid'] else None)
        result['bins'].append(stat)
        fmt = lambda value: f'{value:.4f}' if value is not None else 'NA'
        print(f'{left:g}-{right:g} m: IoU={fmt(stat["iou"])}, '
              f'precision={fmt(stat["precision"])}, '
              f'recall={fmt(stat["recall"])}, '
              f'TP/FP/FN={tp}/{fp}/{fn}')
    total_tp = sum(item['tp'] for item in counts)
    total_fp = sum(item['fp'] for item in counts)
    total_fn = sum(item['fn'] for item in counts)
    result['overall'] = dict(
        tp=total_tp, fp=total_fp, fn=total_fn,
        iou=total_tp / (total_tp + total_fp + total_fn)
        if total_tp + total_fp + total_fn else None,
        precision=total_tp / (total_tp + total_fp)
        if total_tp + total_fp else None,
        recall=total_tp / (total_tp + total_fn)
        if total_tp + total_fn else None)
    print(f'Overall: IoU={fmt(result["overall"]["iou"])}, '
          f'precision={fmt(result["overall"]["precision"])}, '
          f'recall={fmt(result["overall"]["recall"])}')
    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8')
        print(f'Saved {output}')


if __name__ == '__main__':
    main()
