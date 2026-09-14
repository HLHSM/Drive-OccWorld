#!/usr/bin/env python3
"""Adapt a 0.2 m FarmSim checkpoint to a matching 0.1 m model grid.

All resolution-independent tensors are copied directly.  Learned BEV query
and row/column positional embeddings are resized continuously in BEV space,
so the fine-tuning run starts from the same learned metric-space prior rather
than random 0.1 m queries.  The output is a full target-model checkpoint and
can be passed directly to ``tools/train.py --load-from``.
"""

import argparse
import copy
import importlib
import math
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
import sitecustomize  # noqa: F401,E402

import torch
import torch.nn.functional as F
from mmcv import Config
from mmdet3d.models import build_model


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True,
                        help='completed 0.2 m FarmSim checkpoint')
    parser.add_argument('--config', type=Path, required=True,
                        help='0.1 m target model configuration')
    parser.add_argument('--output', type=Path, required=True,
                        help='full, adapted 0.1 m checkpoint path')
    parser.add_argument('--seed', type=int, default=20260912,
                        help='deterministic initializer for absent target tensors')
    return parser.parse_args()


def import_plugin(config_path, cfg):
    if not cfg.get('plugin', False):
        return
    plugin_dir = cfg.get('plugin_dir', str(config_path.parent))
    importlib.import_module(os.path.dirname(plugin_dir).replace('/', '.'))


def resize_bev_embedding(value, target):
    """Bilinearly resize a square learned BEV query grid."""
    source_tokens, channels = value.shape
    target_tokens, target_channels = target.shape
    source_side = math.isqrt(source_tokens)
    target_side = math.isqrt(target_tokens)
    if (source_side * source_side != source_tokens or
            target_side * target_side != target_tokens or
            channels != target_channels):
        return None
    grid = value.t().reshape(1, channels, source_side, source_side)
    grid = F.interpolate(grid, size=(target_side, target_side),
                         mode='bilinear', align_corners=False)
    return grid.reshape(channels, target_tokens).t().to(dtype=target.dtype)


def resize_axis_embedding(value, target):
    """Linearly resize one learned row/column embedding table."""
    if value.ndim != 2 or target.ndim != 2 or value.shape[1] != target.shape[1]:
        return None
    table = value.t().unsqueeze(0)
    table = F.interpolate(table, size=target.shape[0], mode='linear',
                          align_corners=False)
    return table.squeeze(0).t().to(dtype=target.dtype)


def resized_tensor(key, value, target):
    if key.endswith('bev_embedding.weight'):
        return resize_bev_embedding(value, target)
    if key.endswith('positional_encoding.row_embed.weight'):
        return resize_axis_embedding(value, target)
    if key.endswith('positional_encoding.col_embed.weight'):
        return resize_axis_embedding(value, target)
    return None


def main():
    args = parse_args()
    if not args.source.is_file():
        raise FileNotFoundError(args.source)
    if not args.config.is_file():
        raise FileNotFoundError(args.config)

    cfg = Config.fromfile(str(args.config))
    import_plugin(args.config, cfg)
    torch.manual_seed(args.seed)
    model = build_model(cfg.model, train_cfg=cfg.get('train_cfg'),
                        test_cfg=cfg.get('test_cfg'))
    model.init_weights()
    target = model.state_dict()

    checkpoint = torch.load(str(args.source), map_location='cpu')
    source = checkpoint.get('state_dict', checkpoint)
    source = {key[7:] if key.startswith('module.') else key: value
              for key, value in source.items()}
    adapted, resized, omitted = {}, [], []
    for key, value in source.items():
        target_value = target.get(key)
        if target_value is None:
            omitted.append(key)
            continue
        if value.shape == target_value.shape:
            adapted[key] = value
            continue
        value_resized = resized_tensor(key, value, target_value)
        if value_resized is not None:
            adapted[key] = value_resized
            resized.append((key, tuple(value.shape), tuple(target_value.shape)))
        else:
            omitted.append(key)

    initialized = sorted(set(target) - set(adapted))
    target.update(adapted)
    meta = copy.deepcopy(checkpoint.get('meta', {}))
    meta.update(dict(
        adapted_from=str(args.source.resolve()),
        target_config=str(args.config.resolve()),
        adaptation='0.2 m to 0.1 m BEV interpolation with compatible tensor transfer',
        loaded_tensor_count=len(adapted),
        resized_tensor_count=len(resized),
        initialized_tensor_count=len(initialized),
        initialization_seed=args.seed,
    ))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(dict(meta=meta, state_dict=target), str(args.output))

    print(f'wrote {args.output}')
    print(f'copied compatible tensors: {len(adapted) - len(resized)}')
    print(f'interpolated BEV tensors: {len(resized)}')
    for key, source_shape, target_shape in resized:
        print(f'  {key}: {source_shape} -> {target_shape}')
    print(f'target tensors left at initialization: {len(initialized)}')
    print(f'source-only or incompatible tensors omitted: {len(omitted)}')


if __name__ == '__main__':
    main()
