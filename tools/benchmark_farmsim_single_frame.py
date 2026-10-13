#!/usr/bin/env python3
"""Measure batch-one FarmSim occupancy inference without dataset I/O timing."""

import argparse
import importlib
import json
import statistics
import time
from pathlib import Path

import sitecustomize  # noqa: F401  # Repository MMCV/PyTorch compatibility shims.
import torch
from mmcv import Config
from mmcv.parallel import MMDataParallel
from mmcv.runner import load_checkpoint, wrap_fp16_model
from mmdet3d.datasets import build_dataset
from mmdet3d.models import build_model

from projects.mmdet3d_plugin.datasets.builder import build_dataloader
from projects.mmdet3d_plugin.bevformer.apis.test import _batch_size_from_data


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('config')
    parser.add_argument('checkpoint')
    parser.add_argument('--output', required=True)
    parser.add_argument('--samples', type=int, default=10)
    parser.add_argument('--warmup', type=int, default=2)
    args = parser.parse_args()
    if args.samples < 1 or args.warmup < 0:
        parser.error('--samples must be positive and --warmup non-negative')
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        parser.error('Expose exactly one CUDA GPU with CUDA_VISIBLE_DEVICES')

    cfg = Config.fromfile(args.config)
    if cfg.get('plugin', False):
        importlib.import_module(cfg.plugin_dir.rstrip('/').replace('/', '.'))
    cfg.data.test.samples_per_gpu = 1
    cfg.data.test.max_samples = args.samples + args.warmup
    cfg.model.train_cfg = None
    dataset = build_dataset(cfg.data.test)
    loader = build_dataloader(
        dataset, samples_per_gpu=1, workers_per_gpu=0,
        dist=False, shuffle=False,
        nonshuffler_sampler=cfg.data.nonshuffler_sampler)

    model = build_model(cfg.model, test_cfg=cfg.get('test_cfg'))
    model._return_prediction_artifacts = False
    model._profile_test = False
    if cfg.get('fp16', None) is not None:
        wrap_fp16_model(model)
    load_checkpoint(model, args.checkpoint, map_location='cpu')

    # The normal test path computes a GT confusion matrix inside forward_test.
    # Exclude that evaluation-only work while retaining image/BEV/occupancy
    # prediction and the same MMCV batch-one wrapper as tools/test.py.
    model.evaluate_occ = lambda *unused_args, **unused_kwargs: (0, 0, 0, 0)
    model = MMDataParallel(model.cuda().eval(), device_ids=[0])
    torch.cuda.synchronize()
    model_allocated_mib = torch.cuda.memory_allocated() / 2**20
    torch.cuda.reset_peak_memory_stats()

    wall_ms = []
    cuda_ms = []
    with torch.no_grad():
        for index, data in enumerate(loader):
            if _batch_size_from_data(data) != 1:
                raise RuntimeError('Expected exactly one image sample per forward')
            torch.cuda.synchronize()
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)
            start_wall = time.perf_counter()
            start_event.record()
            model(return_loss=False, rescale=True, **data)
            end_event.record()
            torch.cuda.synchronize()
            elapsed_wall_ms = (time.perf_counter() - start_wall) * 1000
            if index >= args.warmup:
                wall_ms.append(elapsed_wall_ms)
                cuda_ms.append(start_event.elapsed_time(end_event))
            if len(wall_ms) >= args.samples:
                break
    if len(wall_ms) != args.samples:
        raise RuntimeError(f'Only measured {len(wall_ms)}/{args.samples} frames')

    summary = {
        'config': str(Path(args.config).resolve()),
        'checkpoint': str(Path(args.checkpoint).resolve()),
        'gpu': torch.cuda.get_device_name(0),
        'batch_size': 1,
        'warmup_frames': args.warmup,
        'measured_frames': len(wall_ms),
        'timing_scope': 'MMDataParallel model forward; excludes dataloader and GT metrics',
        'wall_ms': wall_ms,
        'wall_mean_ms': statistics.mean(wall_ms),
        'wall_median_ms': statistics.median(wall_ms),
        'wall_min_ms': min(wall_ms),
        'wall_max_ms': max(wall_ms),
        'cuda_event_ms': cuda_ms,
        'cuda_mean_ms': statistics.mean(cuda_ms),
        'model_allocated_mib': model_allocated_mib,
        'peak_allocated_mib': torch.cuda.max_memory_allocated() / 2**20,
        'peak_reserved_mib': torch.cuda.max_memory_reserved() / 2**20,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(summary, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
