#!/usr/bin/env python3
"""Prepare a small calibration adaptation run from the ORIGINAL train/val split.

New crop-row evaluation sequences must never be passed as adaptation data.
"""
import argparse
import json
from pathlib import Path
import numpy as np
from mmcv import Config


def subset(source, destination, per_sequence):
    manifest = json.loads(Path(source).read_text())
    sequences = []
    for sequence in manifest['sequences']:
        sequence = dict(sequence)
        ids = sequence['frame_ids']
        selected = np.unique(np.linspace(0, len(ids) - 1,
                                        min(per_sequence, len(ids))).round().astype(int))
        sequence['frame_ids'] = [ids[i] for i in selected]
        sequences.append(sequence)
    manifest['sequences'] = sequences
    destination.write_text(json.dumps(manifest, indent=2) + '\n')
    return {s['scenario_id'] for s in sequences}, sum(len(s['frame_ids']) for s in sequences)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('config', type=Path)
    parser.add_argument('checkpoint', type=Path)
    parser.add_argument('output_dir', type=Path)
    parser.add_argument('--train-manifest', type=Path, default=Path('data/farmsim/splits/train.json'))
    parser.add_argument('--val-manifest', type=Path, default=Path('data/farmsim/splits/val.json'))
    parser.add_argument('--train-per-sequence', type=int, default=8)
    parser.add_argument('--val-per-sequence', type=int, default=4)
    args = parser.parse_args()
    root = args.output_dir.resolve()
    root.mkdir(parents=True, exist_ok=True)
    train_ids, ntrain = subset(args.train_manifest, root / 'train.json', args.train_per_sequence)
    val_ids, nval = subset(args.val_manifest, root / 'val.json', args.val_per_sequence)
    if train_ids & val_ids:
        raise ValueError('adaptation train and validation sequences overlap')
    cfg = Config.fromfile(str(args.config))
    cfg.work_dir = str(root)
    cfg.load_from = str(args.checkpoint.resolve())
    cfg.resume_from = None
    for split in ('train', 'val', 'test'):
        cfg.data[split].camera_geometry = 'grid_world'
        cfg.data[split].ann_file = str(root / ('train.json' if split == 'train' else 'val.json'))
        cfg.data[split].samples_per_gpu = 4
    cfg.data.samples_per_gpu = 4
    cfg.data.workers_per_gpu = 2
    # Full-dataset crop-count files have different indices to these subsets.
    cfg.data.shuffler_sampler = dict(type='DistributedGroupSampler')
    cfg.model.img_backbone.frozen_stages = 4
    cfg.optimizer.lr = 5e-5
    cfg.optimizer_config = dict(type='GradientCumulativeOptimizerHook', cumulative_iters=3,
                                grad_clip=dict(max_norm=35, norm_type=2))
    cfg.lr_config = dict(policy='CosineAnnealing', warmup='linear', warmup_iters=30,
                         warmup_ratio=.3333333333333333, min_lr_ratio=.1)
    cfg.runner = dict(type='EpochBasedRunner', max_epochs=2)
    cfg.total_epochs = 2
    cfg.evaluation = dict(interval=1)
    cfg.checkpoint_config = dict(interval=1, max_keep_ckpts=2)
    cfg.log_config = dict(interval=25, hooks=[dict(type='TextLoggerHook')])
    cfg.gpu_ids = [0]
    cfg.seed = 0
    cfg.farmsim_total_batch_size = 12
    cfg.farmsim_grad_accum_steps = 3
    cfg.dump(str(root / 'grid_calibration.py'))
    audit = dict(train_frames=ntrain, val_frames=nval,
                 training_source=str(args.train_manifest.resolve()),
                 validation_source=str(args.val_manifest.resolve()),
                 checkpoint=str(args.checkpoint.resolve()), seed=0,
                 camera_geometry='grid_world', backbone_frozen=True, epochs=2,
                 evaluation_data_used_for_training=False)
    (root / 'adaptation_plan.json').write_text(json.dumps(audit, indent=2) + '\n')
    print(json.dumps(audit, indent=2))


if __name__ == '__main__':
    main()
