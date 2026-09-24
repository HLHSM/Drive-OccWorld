#!/usr/bin/env python3
"""Run a FarmSim checkpoint on synthetic multi-view real-field images.

This utility is deliberately for qualitative visualisation only.  It reuses a
FarmSim validation sample solely for the camera calibration and the tensor
container expected by Drive-OccWorld.  The input RGB views are supplied by the
user and no validation occupancy label is used in the exported result.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import mmcv
import numpy as np
import torch
from mmcv import Config
from mmcv.parallel import collate
from mmcv.runner import load_checkpoint, wrap_fp16_model
from mmdet3d.datasets import build_dataset
from mmdet3d.models import build_model

import sitecustomize  # noqa: F401  # project MMCV compatibility shims
import projects.mmdet3d_plugin  # noqa: F401  # register project modules
from projects.mmdet3d_plugin.bevformer.apis.test import (
    _prepare_distributed_eval_batch,
)


CAMERA_FILES = ('front_left.png', 'front.png', 'front_right.png')


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--views-dir', type=Path, required=True,
                        help='one subdirectory per case, each containing '
                             'front_left.png, front.png and front_right.png')
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--reference-index', type=int, default=0,
                        help='FarmSim validation sample used only for calibration')
    return parser.parse_args()


def load_view_tensor(case_dir: Path, image_size: tuple[int, int]) -> torch.Tensor:
    """Apply the FarmSim BGR resize/normalisation used at test time."""
    width, height = image_size
    views = []
    for filename in CAMERA_FILES:
        path = case_dir / filename
        if not path.is_file():
            raise FileNotFoundError(f'Missing required camera view: {path}')
        image = mmcv.imread(str(path), flag='color').astype(np.float32)
        image = mmcv.imresize(image, (width, height))
        image = mmcv.imnormalize(
            image, np.array([103.530, 116.280, 123.675]),
            np.ones(3), to_rgb=False)
        views.append(image.transpose(2, 0, 1))
    return torch.from_numpy(np.stack(views)).float()


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available() and args.device.startswith('cuda'):
        raise RuntimeError(f'{args.device} was requested but CUDA is unavailable')
    if not args.config.is_file() or not args.checkpoint.is_file():
        raise FileNotFoundError('config or checkpoint does not exist')

    case_dirs = sorted(path for path in args.views_dir.iterdir() if path.is_dir())
    if not case_dirs:
        raise FileNotFoundError(f'No case directories found in {args.views_dir}')

    cfg = Config.fromfile(str(args.config))
    dataset = build_dataset(cfg.data.test)
    if not 0 <= args.reference_index < len(dataset):
        raise IndexError(f'reference-index must be in [0, {len(dataset) - 1}]')
    if dataset.queue_length != 0 or not dataset.front_only:
        raise RuntimeError('This qualitative utility requires the no-history, '
                           'three-front-camera FarmSim configuration.')

    cfg.model.train_cfg = None
    model = build_model(cfg.model, test_cfg=cfg.get('test_cfg'))
    if cfg.get('fp16') is not None:
        wrap_fp16_model(model)
    load_checkpoint(model, str(args.checkpoint), map_location='cpu')
    device = torch.device(args.device)
    model.to(device).eval()
    # Reuse the standard artifact path, then discard its reference GT before
    # saving.  ``segmentation`` remains only as a shape carrier required by
    # the legacy forward_test implementation.
    model._return_prediction_artifacts = True
    model._prediction_artifact_limit = -1
    model._prediction_artifact_indices = None

    args.output_dir.mkdir(parents=True, exist_ok=True)
    reference = collate([dataset[args.reference_index]], samples_per_gpu=1)
    for case_dir in case_dirs:
        data = _prepare_distributed_eval_batch(reference, device)
        # [B=1, T=1, Ncam=3, C, H, W]; calibration metadata comes from the
        # matching FarmSim front-camera rig and input content is replaced.
        data['img'] = data['img'].clone()
        data['img'][0, 0] = load_view_tensor(case_dir, dataset.image_size).to(device)
        with torch.no_grad():
            result = model(return_loss=False, rescale=True, **data)
        artifacts = result.get('prediction_artifacts', [])
        if len(artifacts) != 1:
            raise RuntimeError(f'{case_dir.name}: expected one prediction artifact, '
                               f'got {len(artifacts)}')
        artifact = artifacts[0]
        prediction = artifact['current_pred']
        # The common preview renderer needs a same-shape validity mask.  Zero
        # values here mean "valid for rendering", never a ground-truth label;
        # the accompanying manifest records that no real occupancy GT exists.
        np.savez_compressed(
            args.output_dir / f'{case_dir.name}.npz',
            current_pred=prediction,
            current_gt=np.zeros_like(prediction, dtype=np.uint8),
            point_cloud_range=artifact['point_cloud_range'],
            num_classes=artifact['num_classes'],
            synthetic_multiview=np.array(True),
        )
        print(f'wrote prediction: {args.output_dir / (case_dir.name + ".npz")}',
              flush=True)

    manifest = {
        'purpose': 'qualitative-only synthetic multi-view real-field inference',
        'checkpoint': str(args.checkpoint.resolve()),
        'config': str(args.config.resolve()),
        'camera_calibration_source': (
            f'FarmSim validation sample index {args.reference_index}; '
            'used only for the three-camera extrinsics/intrinsics'),
        'input_provenance': (
            'left/front/right images were generated or reconstructed from '
            'real forward photographs; they are not synchronised measurements'),
        'ground_truth_occupancy_available': False,
        'cases': [path.name for path in case_dirs],
    }
    (args.output_dir / 'README.json').write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')


if __name__ == '__main__':
    main()
