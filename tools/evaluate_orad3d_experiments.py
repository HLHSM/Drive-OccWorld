#!/usr/bin/env python3
"""Evaluate selected ORAD-3D experiments and summarize occupancy metrics.

Each checkpoint is evaluated on both the requested mixed farm subset and the
official test split.  Raw confusion matrices are kept as pickle files, while
the aggregated semantic and binary IoUs are written to a CSV.
"""

import argparse
import csv
import os
import pickle
import subprocess
import sys
from pathlib import Path

import numpy as np


CLASS_NAMES = (
    'free', 'road', 'safe-road', 'car', 'people', 'water', 'snow',
    'grass-on-road', 'rock',
)
EXPERIMENTS = (
    'orad3d_farmsim_ft_p10_ep1_20260905_203647',
    'orad3d_farmsim_ft_p10_ep4_20260905_102838',
    'orad3d_farmsim_ft_p25_ep1_20260905_204238',
    'orad3d_farmsim_ft_p25_ep4_20260905_104051',
    'orad3d_farmsim_ft_p50_ep1_20260905_205021',
    'orad3d_farmsim_ft_p50_ep4_20260905_105917',
    'orad3d_farmsim_ft_p100_ep1_20260905_210028',
    'orad3d_farmsim_ft_p100_ep4_20260905_112706',
    'orad3d_scratch_p100_ep8_20260905_085453',
)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo-root', type=Path,
                        default=Path(__file__).resolve().parents[1])
    parser.add_argument('--python-bin', default='/home/HL/.conda/envs/dow2/bin/python')
    parser.add_argument('--cuda-visible-devices',
                        default=os.environ.get('CUDA_VISIBLE_DEVICES', '0,1,2,3'))
    parser.add_argument('--num-gpus', type=int,
                        default=int(os.environ.get('NUM_GPUS', '4')))
    parser.add_argument('--batch-size', type=int,
                        default=int(os.environ.get('BATCH_SIZE', '3')))
    parser.add_argument('--orad-root', type=Path,
                        default=Path('/data/HL/ORAD-3D/extracted'))
    parser.add_argument('--output-dir', type=Path,
                        default=Path('work_dirs/orad3d_evaluations'))
    parser.add_argument('--force', action='store_true',
                        help='rerun entries even if the raw pickle already exists')
    return parser.parse_args()


def latest_checkpoint(experiment_dir):
    checkpoints = sorted(
        experiment_dir.glob('epoch_*.pth'),
        key=lambda path: int(path.stem.removeprefix('epoch_')))
    if not checkpoints:
        raise FileNotFoundError(f'No epoch checkpoint in {experiment_dir}')
    return checkpoints[-1]


def aggregate_histogram(value):
    hist = np.asarray(value)
    if hist.ndim < 2 or hist.shape[-1] != hist.shape[-2]:
        raise ValueError(f'Invalid confusion matrix shape: {hist.shape}')
    return hist.reshape(-1, hist.shape[-2], hist.shape[-1]).sum(axis=0)


def metrics_from_pickle(path):
    with path.open('rb') as handle:
        outputs = pickle.load(handle)
    if not isinstance(outputs, dict):
        raise TypeError(f'{path}: expected occupancy result dictionary')
    key = 'hist_for_iou_current'
    if key not in outputs:
        key = 'hist_for_iou'
    if key not in outputs:
        raise KeyError(f'{path}: no current occupancy confusion matrix')

    hist = aggregate_histogram(outputs[key]).astype(np.float64, copy=False)
    if hist.shape != (len(CLASS_NAMES), len(CLASS_NAMES)):
        raise ValueError(f'{path}: expected 9x9 ORAD histogram, got {hist.shape}')
    diagonal = np.diag(hist)
    union = hist.sum(axis=0) + hist.sum(axis=1) - diagonal
    present = union > 0
    iou = np.divide(diagonal, union, out=np.zeros_like(diagonal), where=present)
    occupied_tp = hist[1:, 1:].sum()
    occupied_union = occupied_tp + hist[0, 1:].sum() + hist[1:, 0].sum()
    free_union = hist[0, :].sum() + hist[:, 0].sum() - hist[0, 0]
    occupied_iou = occupied_tp / occupied_union if occupied_union else 0.0
    free_iou = hist[0, 0] / free_union if free_union else 0.0

    result = dict(
        semantic_mIoU=float(iou[present].mean()) if present.any() else 0.0,
        semantic_mIoU_all_classes=float(iou.mean()),
        voxel_accuracy=float(diagonal.sum() / hist.sum()) if hist.sum() else 0.0,
        binary_IoU_free=float(free_iou),
        binary_IoU_occupied=float(occupied_iou),
        binary_mIoU=float((free_iou + occupied_iou) / 2.0),
    )
    result.update({f'IoU_{name}': float(value) for name, value in zip(CLASS_NAMES, iou)})
    return result


def run_evaluation(args, checkpoint, subset_name, manifest, raw_output, log_path):
    command = [
        args.python_bin, '-m', 'torch.distributed.run', '--standalone',
        f'--nproc_per_node={args.num_gpus}', 'tools/test.py',
        'projects/configs/orad3d/orad3d_occ_mono.py', str(checkpoint),
        '--launcher', 'pytorch', '--batch-size', str(args.batch_size),
        '--eval', 'mIoU', '--out', str(raw_output), '--cfg-options',
        f'data.test.data_root={args.orad_root}',
        f'data.test.ann_file={manifest}',
        'data.test.image_size=[512,288]',
        'data.test.queue_length=0',
        'model.future_pred_head.history_queue_length=0',
    ]
    environment = os.environ.copy()
    environment['CUDA_VISIBLE_DEVICES'] = args.cuda_visible_devices
    environment['PYTHONPATH'] = f'{args.repo_root}:{environment.get("PYTHONPATH", "")}'
    with log_path.open('w', encoding='utf-8') as handle:
        subprocess.run(command, cwd=args.repo_root, env=environment,
                       stdout=handle, stderr=subprocess.STDOUT, check=True)
    print(f'completed {checkpoint.parent.name} on {subset_name}')


def main():
    args = parse_args()
    args.repo_root = args.repo_root.resolve()
    args.output_dir = (args.repo_root / args.output_dir).resolve()
    config = args.repo_root / 'projects/configs/orad3d/orad3d_occ_mono.py'
    subsets = (
        ('farm_all', args.repo_root / 'data/orad3d/splits/farm_all.json'),
        ('test', args.repo_root / 'data/orad3d/splits/test.json'),
    )
    if args.num_gpus < 1 or args.batch_size < 1:
        raise ValueError('--num-gpus and --batch-size must be positive')
    if not config.is_file() or not args.orad_root.is_dir():
        raise FileNotFoundError('Missing ORAD config or extracted data root')
    for _, manifest in subsets:
        if not manifest.is_file():
            raise FileNotFoundError(f'Missing evaluation manifest: {manifest}')

    rows = []
    for experiment_name in EXPERIMENTS:
        experiment_dir = args.repo_root / 'work_dirs' / experiment_name
        checkpoint = latest_checkpoint(experiment_dir)
        for subset_name, manifest in subsets:
            result_dir = args.output_dir / experiment_name
            raw_output = result_dir / f'{subset_name}.pkl'
            log_path = result_dir / f'{subset_name}.log'
            result_dir.mkdir(parents=True, exist_ok=True)
            if args.force or not raw_output.is_file():
                run_evaluation(args, checkpoint, subset_name, manifest,
                               raw_output, log_path)
            metrics = metrics_from_pickle(raw_output)
            rows.append(dict(
                experiment=experiment_name,
                checkpoint=str(checkpoint.relative_to(args.repo_root)),
                subset=subset_name,
                manifest=str(manifest.relative_to(args.repo_root)),
                raw_result=str(raw_output.relative_to(args.repo_root)),
                **metrics,
            ))

    fieldnames = (
        'experiment', 'checkpoint', 'subset', 'manifest', 'raw_result',
        'semantic_mIoU', 'semantic_mIoU_all_classes', 'voxel_accuracy',
        *(f'IoU_{name}' for name in CLASS_NAMES),
        'binary_IoU_free', 'binary_IoU_occupied', 'binary_mIoU',
    )
    csv_path = args.output_dir / 'orad3d_eval_summary.csv'
    with csv_path.open('w', encoding='utf-8', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f'wrote {csv_path}')


if __name__ == '__main__':
    main()
