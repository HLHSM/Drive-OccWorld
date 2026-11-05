#!/usr/bin/env python3
"""Render Drive-OccWorld panoptic predictions with FlashOCC's SimData renderer."""
import argparse
import ast
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import sitecustomize  # noqa: F401
import mmcv
import numpy as np
import torch
from mmcv.runner import wrap_fp16_model
from mmdet3d.models import build_model
from PIL import Image, ImageDraw, ImageFont

import projects.mmdet3d_plugin  # noqa: F401
from projects.mmdet3d_plugin.datasets.farmsim_panoptic_dataset import (
    FarmSimPanopticDataset, PANOPTIC_CLASSES, THING_IDS)

FLASH_RENDERER = Path('/data/HL/FlashOCC/tools/visualize_simdata_sequences.py')


def load_flash_renderer(path):
    """Use the exact FlashOCC rendering functions without importing its model plugin."""
    tree = ast.parse(path.read_text(encoding='utf-8'), filename=str(path))
    allowed_names = {'COLORS', 'UNKNOWN', 'FONT'}
    allowed_functions = {'top_view', 'render', 'write_index'}
    selected = []
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(
                isinstance(target, ast.Name) and target.id in allowed_names for target in node.targets):
            selected.append(node)
        elif isinstance(node, ast.FunctionDef) and node.name in allowed_functions:
            selected.append(node)
    namespace = dict(np=np, Image=Image, ImageDraw=ImageDraw, ImageFont=ImageFont,
                     json=json, CLASSES=PANOPTIC_CLASSES, THINGS=THING_IDS)
    exec(compile(ast.Module(body=selected, type_ignores=[]), str(path), 'exec'), namespace)
    return namespace['render'], namespace['write_index']


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--out-dir', required=True)
    parser.add_argument('--sequences', nargs='+', required=True)
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--threads', type=int, default=8)
    parser.add_argument('--max-frames', type=int)
    parser.add_argument('--workers', type=int, default=1,
                        help='CPU inference worker processes; each uses --threads threads')
    parser.add_argument('--shard-index', type=int, default=0, help=argparse.SUPPRESS)
    parser.add_argument('--num-shards', type=int, default=1, help=argparse.SUPPRESS)
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    device = torch.device(args.device)
    out = Path(args.out_dir).resolve()
    out.mkdir(parents=True, exist_ok=True)
    if args.workers > 1:
        if args.num_shards != 1 or args.max_frames is not None or device.type != 'cpu':
            raise ValueError('Parallel mode requires CPU, no frame limit, and no explicit shard')
        _, write_index = load_flash_renderer(FLASH_RENDERER)
        children = []
        for shard in range(args.workers):
            cmd = [sys.executable, str(Path(__file__).resolve()), '--config', args.config,
                   '--checkpoint', args.checkpoint, '--out-dir', str(out),
                   '--sequences', *args.sequences, '--device', args.device,
                   '--threads', str(args.threads), '--num-shards', str(args.workers),
                   '--shard-index', str(shard)]
            handle = (out / f'worker_{shard}.log').open('w')
            children.append((subprocess.Popen(cmd, stdout=handle, stderr=subprocess.STDOUT), handle))
        try:
            while any(process.poll() is None for process, _ in children):
                failures = [(i, process.returncode) for i, (process, _) in enumerate(children)
                            if process.poll() not in (None, 0)]
                if failures:
                    raise RuntimeError(f'Inference worker failed: {failures}')
                write_index(out, args.sequences)
                counts = {seq: len(list((out / seq / 'frames').glob('*.png'))) for seq in args.sequences}
                (out / 'progress.json').write_text(json.dumps(counts, indent=2) + '\n')
                print('frames rendered:', counts, flush=True)
                time.sleep(20)
            failures = [(i, process.returncode) for i, (process, _) in enumerate(children)
                        if process.returncode != 0]
            if failures:
                raise RuntimeError(f'Inference worker failed: {failures}')
        finally:
            for process, handle in children:
                if process.poll() is None:
                    process.terminate()
                process.wait()
                handle.close()
        cfg = mmcv.Config.fromfile(args.config)
        root = Path(cfg.data.test.data_root)
        counts = {}
        for seq in args.sequences:
            expected = {p.stem for p in (root / seq / 'meta').glob('*.json')}
            images = {p.stem for p in (out / seq / 'frames').glob('*.png')}
            predictions = {p.stem for p in (out / seq / 'predictions').glob('*.npz')}
            if images != expected or predictions != expected:
                raise RuntimeError(f'{seq}: missing or extra images/NPZs')
            counts[seq] = len(expected)
        write_index(out, args.sequences)
        (out / 'complete.json').write_text(json.dumps(dict(
            config=str(Path(args.config).resolve()), checkpoint=str(Path(args.checkpoint).resolve()),
            renderer=str(FLASH_RENDERER), frame_count=sum(counts.values()),
            per_sequence=counts, sequences=args.sequences, classes=PANOPTIC_CLASSES,
            device=str(device), coordinate_system='FlashOCC: x forward, y left; arrays [x,y,z]',
            predictions='unmasked pred_occ uint8 and pano_inst uint32'), indent=2) + '\n')
        print('Complete:', counts, flush=True)
        return
    cfg = mmcv.Config.fromfile(args.config)
    manifest = out / 'sequences.json'
    temp_manifest = out / f'.sequences.{os.getpid()}.json'
    temp_manifest.write_text(json.dumps({'data_root': cfg.data.test.data_root,
                                         'sequences': args.sequences}, indent=2) + '\n')
    os.replace(temp_manifest, manifest)
    ds_cfg = dict(cfg.data.test)
    ds_cfg.update(ann_file=str(manifest), sample_stride=1, test_mode=True)
    for key in ('type', 'samples_per_gpu', 'max_samples'):
        ds_cfg.pop(key, None)
    dataset = FarmSimPanopticDataset(**ds_cfg)
    cfg.model.train_cfg = None
    model = build_model(cfg.model, test_cfg=cfg.get('test_cfg'))
    if device.type == 'cuda' and cfg.get('fp16') is not None:
        wrap_fp16_model(model)
    checkpoint = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    state = checkpoint['state_dict']
    state = {k.removeprefix('module.'): v for k, v in state.items()}
    incompatible = model.load_state_dict(state, strict=False)
    # These two weights belong only to the deleted detection decoder path;
    # current-frame BEV extraction uses get_bev_features instead.
    allowed_missing = {'pts_bbox_head.transformer.reference_points.weight',
                       'pts_bbox_head.transformer.reference_points.bias'}
    if set(incompatible.missing_keys) - allowed_missing or incompatible.unexpected_keys:
        raise RuntimeError(f'Unexpected checkpoint mismatch: {incompatible}')
    del state
    print(f"Checkpoint epoch={checkpoint.get('meta', {}).get('epoch')} device={device}; "
          f'{len(dataset)} frames', flush=True)
    del checkpoint
    model.to(device).eval()
    render, write_index = load_flash_renderer(FLASH_RENDERER)
    count = len(dataset) if args.max_frames is None else min(len(dataset), args.max_frames)
    start = time.monotonic()
    processed = 0
    # Interleave sequences so each receives previews early; rotate their order
    # to distribute long and short sequences across workers.
    per_sequence = [[] for _ in dataset.sequences]
    for index, (seq_idx, _) in enumerate(dataset.samples[:count]):
        per_sequence[seq_idx].append(index)
    order = []
    for frame_index in range(max(map(len, per_sequence), default=0)):
        available = [seq for seq in per_sequence if frame_index < len(seq)]
        if available:
            shift = frame_index % len(available)
            order.extend(seq[frame_index] for seq in available[shift:] + available[:shift])
    for index in order[args.shard_index::args.num_shards]:
        seq_idx, frame_idx = dataset.samples[index]
        seq_info = dataset.sequences[seq_idx]
        seq = dataset._sequence_path(seq_info)
        frame = seq_info['frame_ids'][frame_idx]
        dest = out / seq_info['path']
        png = dest / 'frames' / f'{frame}.png'
        npz = dest / 'predictions' / f'{frame}.npz'
        if png.is_file() and npz.is_file():
            continue
        sample = dataset[index]
        img = sample['img'].data.unsqueeze(0).to(device)
        metas = [sample['img_metas'].data]
        with torch.inference_mode():
            outputs, _ = model._forward_panoptic(img, metas)
            sem_zyx = outputs['semantic'].argmax(1)
            instance_zyx = model.future_pred_head.instance_decoder.predict(outputs['instance'], sem_zyx)
        pred = sem_zyx[0].permute(2, 1, 0).cpu().numpy().astype(np.uint8)
        pred_inst = instance_zyx[0].permute(2, 1, 0).cpu().numpy().astype(np.uint32)
        gt = sample['segmentation'][0].numpy().astype(np.uint8)
        gt_inst = sample['instance'][0].numpy().astype(np.uint32)
        if gt.shape != pred.shape or gt_inst.shape != pred_inst.shape:
            raise ValueError(f'{seq}/{frame}: prediction/GT shape mismatch')
        # Drive-OccWorld stores UE x-forward/y-right. FlashOCC's load_target
        # flips y before its top_view function; transform both GT and prediction.
        gt, gt_inst = gt[:, ::-1, :].copy(), gt_inst[:, ::-1, :].copy()
        pred, pred_inst = pred[:, ::-1, :].copy(), pred_inst[:, ::-1, :].copy()
        gt_inst[gt == 255] = 0
        png.parent.mkdir(parents=True, exist_ok=True)
        npz.parent.mkdir(parents=True, exist_ok=True)
        render(seq, frame, gt, gt_inst, pred, pred_inst, png)
        np.savez_compressed(npz, pred_occ=pred, pano_inst=pred_inst)
        processed += 1
        elapsed = time.monotonic() - start
        print(f'{index+1}/{count} {seq.name}/{frame} elapsed={elapsed:.1f}s '
              f'processed={processed}', flush=True)
        if args.num_shards == 1 and processed % 10 == 0:
            write_index(out, args.sequences)
    if args.num_shards == 1:
        write_index(out, args.sequences)
    if args.num_shards == 1 and args.max_frames is None:
        (out / 'complete.json').write_text(json.dumps(dict(
            config=str(Path(args.config).resolve()), checkpoint=str(Path(args.checkpoint).resolve()),
            renderer=str(FLASH_RENDERER), frame_count=len(dataset), sequences=args.sequences,
            classes=PANOPTIC_CLASSES, device=str(device),
            coordinate_system='FlashOCC: x forward, y left; arrays [x,y,z]',
            predictions='unmasked pred_occ uint8 and pano_inst uint32'), indent=2) + '\n')


if __name__ == '__main__':
    main()
