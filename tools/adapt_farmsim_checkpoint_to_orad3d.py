#!/usr/bin/env python3
"""Filter a FarmSim checkpoint for direct ORAD occupancy-head replacement."""

import argparse
import copy
import os
import sys
from pathlib import Path

# This tool is often invoked through an absolute path from another working
# directory.  Make the repository's startup compatibility module available
# before importing MMDetection/MMCV.
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
import sitecustomize  # noqa: F401,E402

import torch
from mmcv import Config
from mmdet3d.models import build_model


CAMERA_EMBED_KEY = 'pts_bbox_head.transformer.cams_embeds'


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True,
                        help='trained FarmSim checkpoint')
    parser.add_argument('--config', type=Path, required=True,
                        help='target ORAD model config')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--center-camera-index', type=int, default=1,
                        help='FarmSim center-front camera embedding index')
    parser.add_argument('--seed', type=int, default=20260904,
                        help='deterministic initializer for replaced ORAD layers')
    return parser.parse_args()


def import_plugin(config_path, cfg):
    if not cfg.get('plugin', False):
        return
    import importlib
    plugin_dir = cfg.get('plugin_dir', str(config_path.parent))
    module_dir = os.path.dirname(plugin_dir).replace('/', '.')
    importlib.import_module(module_dir)


def main():
    args = parse_args()
    if not args.source.is_file():
        raise FileNotFoundError(args.source)
    cfg = Config.fromfile(str(args.config))
    import_plugin(args.config, cfg)
    torch.manual_seed(args.seed)
    model = build_model(
        cfg.model,
        train_cfg=cfg.get('train_cfg'),
        test_cfg=cfg.get('test_cfg'))
    model.init_weights()
    target = model.state_dict()

    checkpoint = torch.load(str(args.source), map_location='cpu')
    source = checkpoint.get('state_dict', checkpoint)
    source = {key[7:] if key.startswith('module.') else key: value
              for key, value in source.items()}
    adapted, shape_mismatch, missing_target = {}, [], []
    for key, value in source.items():
        if key not in target:
            missing_target.append(key)
            continue
        if target[key].shape == value.shape:
            adapted[key] = value
            continue
        if key == CAMERA_EMBED_KEY:
            index = args.center_camera_index
            if (value.ndim == 2 and target[key].ndim == 2 and
                    target[key].shape[0] == 1 and
                    value.shape[1:] == target[key].shape[1:] and
                    0 <= index < value.shape[0]):
                adapted[key] = value[index:index + 1].clone()
                continue
        shape_mismatch.append(
            (key, tuple(value.shape), tuple(target[key].shape)))

    never_initialized = sorted(set(target) - set(adapted))
    meta = copy.deepcopy(checkpoint.get('meta', {}))
    # Store the complete ORAD model state.  Thus the direct replacement head
    # is initialized exactly once and every 10/25/50/100% fine-tune starts
    # from identical weights; no runtime conversion module is introduced.
    target.update(adapted)
    meta.update(dict(
        adapted_from=str(args.source.resolve()),
        target_config=str(args.config),
        adaptation='direct ORAD head replacement; incompatible tensors omitted',
        loaded_tensor_count=len(adapted),
        initialized_tensor_count=len(never_initialized),
        initialization_seed=args.seed,
    ))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(dict(meta=meta, state_dict=target), str(args.output))

    print(f'wrote {args.output}')
    print(f'loaded compatible tensors: {len(adapted)}')
    print(f'target tensors left at ORAD initialization: {len(never_initialized)}')
    print('shape-mismatched tensors replaced by ORAD modules:')
    for key, source_shape, target_shape in shape_mismatch:
        print(f'  {key}: {source_shape} -> {target_shape}')
    if missing_target:
        print(f'source-only tensors ignored: {len(missing_target)}')


if __name__ == '__main__':
    main()
