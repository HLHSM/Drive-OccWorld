#!/usr/bin/env python3
"""Time official FarmSim baselines on one GPU, without GT evaluation or I/O.

Run from the relevant official repository with its own Python environment and
that repository on PYTHONPATH. Add this workspace after interpreter startup to
avoid importing the unrelated root sitecustomize into older MMCV environments.
"""
import argparse
import importlib
import json
from pathlib import Path
import statistics
import sys
import time
import types

import torch
from mmcv import Config
from mmcv.parallel import MMDataParallel
from mmcv.runner import load_checkpoint, wrap_fp16_model

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(REPO_ROOT))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('method', choices=('surroundocc', 'cotr', 'sparseocc'))
    parser.add_argument('config')
    parser.add_argument('checkpoint')
    parser.add_argument('--output', required=True)
    parser.add_argument('--warmup', type=int, default=10)
    parser.add_argument('--samples', type=int, default=50)
    args = parser.parse_args()
    if args.warmup < 0 or args.samples < 1 or torch.cuda.device_count() != 1:
        parser.error('positive samples, non-negative warmup, and one GPU required')

    cfg = Config.fromfile(args.config)
    if args.method == 'surroundocc':
        importlib.import_module('projects.mmdet3d_plugin')
        from projects.mmdet3d_plugin.datasets.builder import build_dataloader
    elif args.method == 'sparseocc':
        importlib.import_module('models')
        importlib.import_module('loaders')
        from loaders.builder import build_dataloader
    else:
        from mmdet3d.datasets import build_dataloader
    from mmdet3d.datasets import build_dataset
    from mmdet3d.models import build_model

    cfg.data.test.test_mode = True
    cfg.data.test.pop('samples_per_gpu', None)
    cfg.data.test.smoke_num_samples = args.warmup + args.samples
    cfg.model.train_cfg = None
    cfg.model.pretrained = None
    dataset = build_dataset(cfg.data.test)
    loader = build_dataloader(dataset, samples_per_gpu=1, workers_per_gpu=0,
                              dist=False, shuffle=False)
    model = build_model(cfg.model, test_cfg=cfg.get('test_cfg'))
    if cfg.get('fp16') is not None:
        wrap_fp16_model(model)
    load_checkpoint(model, args.checkpoint, map_location='cpu')

    # Preserve each model's network and native input wrapper but omit metric
    # calculation and result CPU serialization, as in the AgriOcc benchmark.
    if args.method in ('surroundocc', 'sparseocc'):
        def forward_without_evaluation(self, img_metas, img=None, **kwargs):
            return self.simple_test(img_metas, img)
        model.forward_test = types.MethodType(forward_without_evaluation, model)
        if args.method == 'sparseocc':
            # Exclude mask-to-semantic postprocessing, matching the raw-logit
            # endpoint used for the other dense and sparse occupancy heads.
            def sparse_head_logits(self, x, img_metas, rescale=False):
                return self.pts_bbox_head(x, img_metas)
            model.simple_test_pts = types.MethodType(sparse_head_logits, model)
    else:
        def simple_test_logits(self, points, img_metas, img=None, **kwargs):
            return self.forward_dummy(points=points, img_metas=img_metas,
                                      img_inputs=img, **kwargs)
        model.simple_test = types.MethodType(simple_test_logits, model)

    model = MMDataParallel(model.cuda().eval(), device_ids=[0])
    torch.cuda.synchronize()
    wall_ms = []
    with torch.no_grad():
        for index, data in enumerate(loader):
            torch.cuda.synchronize()
            start = time.perf_counter()
            model(return_loss=False, rescale=True, **data)
            torch.cuda.synchronize()
            elapsed = (time.perf_counter() - start) * 1000
            if index >= args.warmup:
                wall_ms.append(elapsed)
            if len(wall_ms) == args.samples:
                break
    if len(wall_ms) != args.samples:
        raise RuntimeError(f'Only measured {len(wall_ms)}/{args.samples} frames')
    summary = dict(method=args.method, config=str(Path(args.config).resolve()),
                   checkpoint=str(Path(args.checkpoint).resolve()),
                   gpu=torch.cuda.get_device_name(0), batch_size=1,
                   fp16_configured=cfg.get('fp16') is not None,
                   warmup_frames=args.warmup, measured_frames=len(wall_ms),
                   timing_scope='MMDataParallel model forward; excludes dataloader, GT metrics and result CPU serialization',
                   wall_ms=wall_ms, wall_mean_ms=statistics.mean(wall_ms),
                   wall_median_ms=statistics.median(wall_ms),
                   wall_min_ms=min(wall_ms), wall_max_ms=max(wall_ms))
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(summary, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
