# ---------------------------------------------
# Copyright (c) OpenMMLab. All rights reserved.
# ---------------------------------------------
#  Modified by Zhiqi Li
# ---------------------------------------------
 
from __future__ import division

import argparse
import copy
import mmcv
import os
import time
import torch
import warnings
from mmcv import Config, DictAction
from mmcv.runner import get_dist_info, init_dist
from os import path as osp

# ``dow2`` uses lightweight MMCV for occupancy-only training.  Importing this
# repository hook before MMDetection keeps unrelated generic deformable-attn
# registrations from requiring mmcv-full at module-import time.
import sitecustomize  # noqa: F401

from mmdet import __version__ as mmdet_version
from mmdet3d import __version__ as mmdet3d_version
#from mmdet3d.apis import train_model

from mmdet3d.datasets import build_dataset
from mmdet3d.models import build_model
from mmdet3d.utils import collect_env, get_root_logger
from mmdet.apis import set_random_seed
from mmseg import __version__ as mmseg_version

from mmcv.utils import TORCH_VERSION, digit_version


def parse_args():
    parser = argparse.ArgumentParser(description='Train a detector')
    parser.add_argument('config', help='train config file path')
    parser.add_argument('--work-dir', help='the dir to save logs and models')
    parser.add_argument(
        '--load-from', help='checkpoint used to initialize model weights')
    parser.add_argument(
        '--resume-from', help='the checkpoint file to resume from')
    parser.add_argument(
        '--no-validate',
        action='store_true',
        help='whether not to evaluate the checkpoint during training')
    farmsim = parser.add_argument_group('FarmSim training options')
    farmsim.add_argument('--data-root', help='FarmSim dataset root directory')
    farmsim.add_argument('--train-ann-file',
                         help='FarmSim training manifest JSON')
    farmsim.add_argument('--val-ann-file',
                         help='FarmSim validation/test manifest JSON')
    farmsim.add_argument('--workers-per-gpu', type=int,
                         help='FarmSim DataLoader workers per GPU')
    farmsim.add_argument('--batch-size', type=int,
                         help='per-GPU batch size for train/val/test')
    farmsim.add_argument('--image-width', type=int,
                         help='FarmSim input image width')
    farmsim.add_argument('--image-height', type=int,
                         help='FarmSim input image height')
    farmsim.add_argument('--epochs', type=int, help='number of training epochs')
    farmsim.add_argument('--history-frames', type=int,
                         help='number of historical frames, excluding current')
    farmsim.add_argument('--predict-future-occ', type=int, choices=(0, 1),
                         help='enable future occupancy prediction')
    farmsim.add_argument('--future-occ-steps', type=int, default=0,
                         help='future occupancy steps when enabled')
    farmsim.add_argument('--predict-future-traj', type=int, choices=(0, 1),
                         help='enable trajectory prediction')
    farmsim.add_argument('--future-traj-steps', type=int, default=6,
                         help='future trajectory steps when enabled')
    farmsim.add_argument('--use-fp16', type=int, choices=(0, 1),
                         help='enable AMP FP16 with dynamic loss scaling')
    farmsim.add_argument('--use-crop-gap-refinement', type=int, choices=(0, 1),
                         help='enable crop/free boundary and gap refinement')
    farmsim.add_argument('--crop-gap-boundary-loss-weight', type=float,
                         default=0.5,
                         help='weight of crop/free 3D boundary supervision')
    farmsim.add_argument('--crop-gap-free-loss-weight', type=float,
                         default=0.25,
                         help='weight of near-crop free-gap supervision')
    farmsim.add_argument('--crop-gap-alpha', type=float, default=3.0,
                         help='maximum near-crop free-gap loss multiplier')
    farmsim.add_argument('--crop-gap-sigma', type=float, default=1.5,
                         help='decay distance in voxels for free-gap weighting')
    farmsim.add_argument('--crop-gap-radius', type=int, default=4,
                         help='maximum voxel distance considered a crop gap')
    farmsim.add_argument('--use-selective-c2f', type=int, choices=(0, 1),
                         help='enable selective 2x2 BEV subquery refinement')
    farmsim.add_argument('--c2f-active-ratio', type=float, default=0.25,
                         help='fraction of uncertain crop/free BEV cells refined')
    farmsim.add_argument('--c2f-channels', type=int, default=128,
                         help='hidden width of the selective C2F subquery decoder')
    farmsim.add_argument('--use-dual-hardness-refinement', type=int,
                         choices=(0, 1),
                         help='enable ADHR training-only agricultural hard-voxel refinement')
    farmsim.add_argument('--dual-hardness-active-ratio', type=float, default=0.04,
                         help='fraction of voxel positions selected by ADHR')
    farmsim.add_argument('--dual-hardness-gap-ratio', type=float, default=0.5,
                         help='ADHR selected-voxel quota reserved for crop/free gaps')
    farmsim.add_argument('--dual-hardness-channels', type=int, default=128,
                         help='hidden width of the ADHR voxel refiner')
    farmsim.add_argument('--dual-hardness-local-scale', type=float, default=0.25,
                         help='ADHR local GT anisotropy weighting scale')
    farmsim.add_argument('--dual-hardness-gap-boost', type=float, default=0.5,
                         help='ADHR crop/free gap weighting boost')
    farmsim.add_argument('--dual-hardness-loss-weight', type=float, default=0.5,
                         help='ADHR selected-voxel refinement loss weight')
    farmsim.add_argument('--dual-hardness-distill-weight', type=float, default=0.1,
                         help='ADHR EMA-teacher distillation loss weight')
    farmsim.add_argument('--dual-hardness-ema-decay', type=float, default=0.99,
                         help='ADHR EMA-teacher decay')
    farmsim.add_argument('--use-gap-residual-refiner', type=int, choices=(0, 1),
                         help='enable end-to-end BEV/logit gap residual refiner')
    farmsim.add_argument('--gap-refiner-channels', type=int, default=24,
                         help='hidden width of the anisotropic 3D gap refiner')
    farmsim.add_argument('--gap-refiner-blocks', type=int, default=3,
                         help='number of anisotropic depthwise 3D refiner blocks')
    farmsim.add_argument('--gap-refiner-coarse-loss-weight', type=float,
                         default=0.15,
                         help='deep-supervision weight for pre-refinement logits')
    farmsim.add_argument('--gap-refiner-boundary-loss-weight', type=float,
                         default=0.25,
                         help='crop/free boundary-gate supervision weight')
    farmsim.add_argument('--gap-refiner-gap-loss-weight', type=float,
                         default=0.5,
                         help='near-crop free-gap preservation loss weight')
    farmsim.add_argument('--gap-refiner-crop-loss-weight', type=float,
                         default=0.5,
                         help='near-free crop-preservation loss weight')
    farmsim.add_argument('--gap-refiner-use-bev-feature', type=int,
                         choices=(0, 1), default=1,
                         help='use ref_bev in GapRef; set 0 for logits-only diagnosis')
    farmsim.add_argument('--gap-refiner-use-image-features', type=int,
                         choices=(0, 1), default=0,
                         help='inject sparse current-image FPN evidence into GapRef')
    farmsim.add_argument('--gap-refiner-image-active-ratio', type=float,
                         default=0.08,
                         help='fraction of BEV cells selected for image evidence')
    farmsim.add_argument('--gap-refiner-image-channels', type=int, default=24,
                         help='projected FPN width of the GapRef image branch')
    farmsim.add_argument('--gap-refiner-image-levels', type=int, default=2,
                         help='number of highest-resolution FPN levels sampled')
    farmsim.add_argument('--gap-refiner-image-crop-ratio', type=float,
                         default=0.5,
                         help='selected-cell quota for confident crop regions')
    farmsim.add_argument('--freeze-gap-refiner-base', action='store_true',
                         help='freeze every non-GapRef parameter for diagnostic training')
    farmsim.add_argument('--use-nearfar-bev', type=int, choices=(0, 1),
                         help='enable deterministic near-far sparse BEV encoding')
    farmsim.add_argument('--nearfar-near-ratio', type=float, default=0.6,
                         help='full-attention fraction along the near BEV axis')
    farmsim.add_argument('--nearfar-far-stride', type=int, default=2,
                         help='regular far-field BEV sampling stride')
    farmsim.add_argument('--disable-temporal-self-attention', type=int,
                         choices=(0, 1),
                         help='remove temporal self-attention when --history-frames=0')
    farmsim.add_argument('--use-r50-image-encoder', type=int, choices=(0, 1),
                         help='use the R50 image backbone; defaults load-from to '
                              'pretrained/r50_fcos3d_pretrain.pth when omitted')
    farmsim.add_argument('--use-gvad-attention', type=int, choices=(0, 1),
                         help='replace no-history TSA with geometry-visible anchor deformable attention')
    farmsim.add_argument('--gvad-use-visibility', type=int, choices=(0, 1),
                         default=1,
                         help='use calibrated BEV projection visibility for GVAD anchors')
    farmsim.add_argument('--gvad-use-local-deformable', type=int,
                         choices=(0, 1), default=1,
                         help='keep the local deformable sampling path in GVAD')
    farmsim.add_argument('--gvad-num-heads', type=int, default=8,
                         help='head count shared by GVAD local and anchor attention')
    farmsim.add_argument('--gvad-anchor-grid-height', type=int, default=4,
                         help='number of GVAD anchor tiles along BEV height')
    farmsim.add_argument('--gvad-anchor-grid-width', type=int, default=8,
                         help='number of GVAD anchor tiles along BEV width')
    farmsim.add_argument('--use-directional-decay-retention', type=int,
                         choices=(0, 1),
                         help='replace no-history TSA with directional decay and selective local retention')
    farmsim.add_argument('--ddsr-retention-radius', type=int, default=15,
                         help='one-sided spatial context radius for directional retention')
    farmsim.add_argument('--ddsr-local-dilation', type=int, default=3,
                         help='dilation of the selective long local kernel')
    farmsim.add_argument('--use-efficient-baseline', type=int, choices=(0, 1),
                         help='use the 2-level FPN / 4-layer / 4-point BEV baseline')
    farmsim.add_argument(
        '--total-batch-size', type=int,
        help='target effective global batch; train.py derives accumulation '
             'from batch-size times num-gpus')
    farmsim.add_argument(
        '--grad-accum-steps', type=int,
        help='legacy manual gradient accumulation override; do not combine '
             'with a conflicting --total-batch-size')
    farmsim.add_argument('--num-gpus', type=int,
                         help='number of torchrun processes expected by FarmSim')
    group_gpus = parser.add_mutually_exclusive_group()
    group_gpus.add_argument(
        '--gpus',
        type=int,
        help='number of gpus to use '
        '(only applicable to non-distributed training)')
    group_gpus.add_argument(
        '--gpu-ids',
        type=int,
        nargs='+',
        help='ids of gpus to use '
        '(only applicable to non-distributed training)')
    parser.add_argument('--seed', type=int, default=0, help='random seed')
    parser.add_argument(
        '--deterministic',
        action='store_true',
        help='whether to set deterministic options for CUDNN backend.')
    parser.add_argument(
        '--options',
        nargs='+',
        action=DictAction,
        help='override some settings in the used config, the key-value pair '
        'in xxx=yyy format will be merged into config file (deprecate), '
        'change to --cfg-options instead.')
    parser.add_argument(
        '--cfg-options',
        nargs='+',
        action=DictAction,
        help='override some settings in the used config, the key-value pair '
        'in xxx=yyy format will be merged into config file. If the value to '
        'be overwritten is a list, it should be like key="[a,b]" or key=a,b '
        'It also allows nested list/tuple values, e.g. key="[(a,b),(c,d)]" '
        'Note that the quotation marks are necessary and that no white space '
        'is allowed.')
    parser.add_argument(
        '--launcher',
        choices=['none', 'pytorch', 'slurm', 'mpi'],
        default='none',
        help='job launcher')
    # PyTorch <=1.x launch used ``--local_rank``; current ``torchrun`` uses
    # ``LOCAL_RANK`` and some compatibility launchers pass ``--local-rank``.
    parser.add_argument('--local_rank', '--local-rank', dest='local_rank',
                        type=int, default=0)
    parser.add_argument(
        '--autoscale-lr',
        action='store_true',
        help='automatically scale lr with the number of gpus')
    args = parser.parse_args()
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)

    if args.options and args.cfg_options:
        raise ValueError(
            '--options and --cfg-options cannot be both specified, '
            '--options is deprecated in favor of --cfg-options')
    if args.options:
        warnings.warn('--options is deprecated in favor of --cfg-options')
        args.cfg_options = args.options

    return args


def apply_farmsim_options(cfg, args):
    """Validate FarmSim CLI arguments and merge their config overrides."""
    values = {
        '--batch-size': args.batch_size,
        '--image-width': args.image_width,
        '--image-height': args.image_height,
        '--epochs': args.epochs,
        '--history-frames': args.history_frames,
        '--total-batch-size': args.total_batch_size,
        '--grad-accum-steps': args.grad_accum_steps,
    }
    for name, value in values.items():
        minimum = 0 if name == '--history-frames' else 1
        if value is not None and value < minimum:
            qualifier = 'non-negative' if minimum == 0 else 'positive'
            raise ValueError(f'{name} must be {qualifier}, got {value}.')
    if args.future_occ_steps < 0 or args.future_traj_steps < 0:
        raise ValueError('future occupancy and trajectory step counts cannot be negative.')
    if args.crop_gap_boundary_loss_weight < 0:
        raise ValueError('--crop-gap-boundary-loss-weight must be non-negative.')
    if args.crop_gap_free_loss_weight < 0:
        raise ValueError('--crop-gap-free-loss-weight must be non-negative.')
    if args.crop_gap_alpha < 0:
        raise ValueError('--crop-gap-alpha must be non-negative.')
    if args.crop_gap_sigma <= 0:
        raise ValueError('--crop-gap-sigma must be positive.')
    if args.crop_gap_radius < 1:
        raise ValueError('--crop-gap-radius must be at least 1.')
    if not 0.0 < args.c2f_active_ratio <= 1.0:
        raise ValueError('--c2f-active-ratio must be in (0, 1].')
    if args.c2f_channels < 8:
        raise ValueError('--c2f-channels must be at least 8.')
    if not 0.0 < args.dual_hardness_active_ratio <= 1.0:
        raise ValueError('--dual-hardness-active-ratio must be in (0, 1].')
    if not 0.0 <= args.dual_hardness_gap_ratio <= 1.0:
        raise ValueError('--dual-hardness-gap-ratio must be in [0, 1].')
    if args.dual_hardness_channels < 8:
        raise ValueError('--dual-hardness-channels must be at least 8.')
    for name, value in (
            ('--dual-hardness-local-scale', args.dual_hardness_local_scale),
            ('--dual-hardness-gap-boost', args.dual_hardness_gap_boost),
            ('--dual-hardness-loss-weight', args.dual_hardness_loss_weight),
            ('--dual-hardness-distill-weight', args.dual_hardness_distill_weight)):
        if value < 0:
            raise ValueError(f'{name} must be non-negative.')
    if not 0.0 <= args.dual_hardness_ema_decay < 1.0:
        raise ValueError('--dual-hardness-ema-decay must be in [0, 1).')
    if args.gap_refiner_channels < 8:
        raise ValueError('--gap-refiner-channels must be at least 8.')
    if args.gap_refiner_blocks < 1:
        raise ValueError('--gap-refiner-blocks must be at least 1.')
    if not 0.0 < args.gap_refiner_image_active_ratio <= 1.0:
        raise ValueError('--gap-refiner-image-active-ratio must be in (0, 1].')
    if args.gap_refiner_image_channels < 8:
        raise ValueError('--gap-refiner-image-channels must be at least 8.')
    if args.gap_refiner_image_levels < 1:
        raise ValueError('--gap-refiner-image-levels must be at least 1.')
    if not 0.0 <= args.gap_refiner_image_crop_ratio <= 1.0:
        raise ValueError('--gap-refiner-image-crop-ratio must be in [0, 1].')
    for name, value in (
            ('--gap-refiner-coarse-loss-weight', args.gap_refiner_coarse_loss_weight),
            ('--gap-refiner-boundary-loss-weight', args.gap_refiner_boundary_loss_weight),
            ('--gap-refiner-gap-loss-weight', args.gap_refiner_gap_loss_weight),
            ('--gap-refiner-crop-loss-weight', args.gap_refiner_crop_loss_weight)):
        if value < 0:
            raise ValueError(f'{name} must be non-negative.')
    if not 0.0 < args.nearfar_near_ratio <= 1.0:
        raise ValueError('--nearfar-near-ratio must be in (0, 1].')
    if args.nearfar_far_stride < 2:
        raise ValueError('--nearfar-far-stride must be at least 2.')
    if (args.disable_temporal_self_attention == 1 and
            args.history_frames != 0):
        raise ValueError('--disable-temporal-self-attention requires '
                         '--history-frames=0, because it removes the only '
                         'path that reads prev_bev.')
    if args.use_r50_image_encoder == 1 and args.use_efficient_baseline == 1:
        raise ValueError('--use-r50-image-encoder and --use-efficient-baseline '
                         'cannot be combined: the latter changes the FPN and '
                         'BEV encoder in addition to using R50.')
    spatial_mixer_count = sum(flag == 1 for flag in (
        args.disable_temporal_self_attention,
        args.use_gvad_attention,
        args.use_directional_decay_retention))
    if spatial_mixer_count > 1:
        raise ValueError('Choose only one of --disable-temporal-self-attention, '
                         '--use-gvad-attention, and '
                         '--use-directional-decay-retention.')
    if ((args.use_gvad_attention == 1 or
         args.use_directional_decay_retention == 1) and
            args.history_frames != 0):
        raise ValueError('The no-history BEV spatial mixers require '
                         '--history-frames=0; otherwise they would discard '
                         'the available prev_bev.')
    if args.use_gvad_attention == 1:
        if args.gvad_num_heads < 1 or 256 % args.gvad_num_heads:
            raise ValueError('--gvad-num-heads must be a positive divisor of 256.')
        if (args.gvad_anchor_grid_height < 1 or
                args.gvad_anchor_grid_width < 1):
            raise ValueError('GVAD anchor grid dimensions must be positive.')
        if not (args.gvad_use_visibility or args.gvad_use_local_deformable):
            raise ValueError('GVAD requires visibility anchors or the local '
                             'deformable path.')
    if args.use_directional_decay_retention == 1:
        if args.ddsr_retention_radius < 2:
            raise ValueError('--ddsr-retention-radius must be at least 2.')
        if args.ddsr_local_dilation < 1:
            raise ValueError('--ddsr-local-dilation must be positive.')
    if (args.use_directional_decay_retention == 1 and
            args.use_nearfar_bev == 1):
        raise ValueError('Directional decay retention requires a dense BEV '
                         'grid and cannot be combined with --use-nearfar-bev=1.')
    if args.data_root is not None and not osp.isdir(args.data_root):
        raise FileNotFoundError(
            f'FarmSim --data-root does not exist: {args.data_root}')
    if args.predict_future_occ == 1 and args.future_occ_steps < 1:
        raise ValueError('--future-occ-steps must be at least 1 when '
                         '--predict-future-occ=1.')
    if args.predict_future_traj == 1 and args.future_traj_steps < 1:
        raise ValueError('--future-traj-steps must be at least 1 when '
                         '--predict-future-traj=1.')
    if args.num_gpus is not None:
        if args.num_gpus < 1:
            raise ValueError('--num-gpus must be positive.')
        visible = os.environ.get('CUDA_VISIBLE_DEVICES', '')
        visible_ids = [item.strip() for item in visible.split(',') if item.strip()]
        if len(visible_ids) != args.num_gpus:
            raise ValueError(
                f'--num-gpus={args.num_gpus} requires exactly that many '
                f'CUDA_VISIBLE_DEVICES entries, got {visible!r}.')

    grad_accum_steps = args.grad_accum_steps
    if args.total_batch_size is not None:
        if args.batch_size is None or args.num_gpus is None:
            raise ValueError('--total-batch-size requires both --batch-size '
                             'and --num-gpus.')
        micro_global_batch = args.batch_size * args.num_gpus
        if args.total_batch_size < micro_global_batch:
            raise ValueError(
                '--total-batch-size must be at least one unaccumulated '
                f'global batch ({micro_global_batch}), got '
                f'{args.total_batch_size}.')
        if args.total_batch_size % micro_global_batch:
            raise ValueError(
                '--total-batch-size must be divisible by '
                f'--batch-size × --num-gpus ({args.batch_size} × '
                f'{args.num_gpus} = {micro_global_batch}), got '
                f'{args.total_batch_size}.')
        derived_steps = args.total_batch_size // micro_global_batch
        if grad_accum_steps is not None and grad_accum_steps != derived_steps:
            raise ValueError(
                '--grad-accum-steps conflicts with --total-batch-size: '
                f'expected {derived_steps}, got {grad_accum_steps}.')
        grad_accum_steps = derived_steps

    has_farmsim_option = any(value is not None for value in (
        args.data_root, args.batch_size, args.image_width, args.image_height,
        args.epochs, args.history_frames, args.predict_future_occ,
        args.predict_future_traj, args.use_fp16, args.use_crop_gap_refinement,
        args.use_selective_c2f, args.use_dual_hardness_refinement,
        args.use_gap_residual_refiner,
        args.use_nearfar_bev,
        args.disable_temporal_self_attention,
        args.use_r50_image_encoder,
        args.use_gvad_attention,
        args.use_directional_decay_retention,
        args.use_efficient_baseline,
        args.num_gpus, args.train_ann_file, args.val_ann_file,
        args.workers_per_gpu, args.total_batch_size, args.grad_accum_steps))
    if not has_farmsim_option:
        return

    occ_steps = args.future_occ_steps if args.predict_future_occ else 0
    trajectory_enabled = bool(args.predict_future_traj)
    overrides = {}
    if args.batch_size is not None:
        overrides.update({
            'data.samples_per_gpu': args.batch_size,
            'data.val.samples_per_gpu': args.batch_size,
            'data.test.samples_per_gpu': args.batch_size,
        })
    if args.image_width is not None and args.image_height is not None:
        image_size = (args.image_width, args.image_height)
        overrides.update({
            'data.train.image_size': image_size,
            'data.val.image_size': image_size,
            'data.test.image_size': image_size,
        })
    elif args.image_width is not None or args.image_height is not None:
        raise ValueError('--image-width and --image-height must be provided together.')
    if args.data_root is not None:
        overrides.update({
            'data.train.data_root': args.data_root,
            'data.val.data_root': args.data_root,
            'data.test.data_root': args.data_root,
        })
    if args.train_ann_file is not None:
        if not osp.isfile(args.train_ann_file):
            raise FileNotFoundError(
                f'FarmSim --train-ann-file does not exist: {args.train_ann_file}')
        overrides['data.train.ann_file'] = args.train_ann_file
    if args.val_ann_file is not None:
        if not osp.isfile(args.val_ann_file):
            raise FileNotFoundError(
                f'FarmSim --val-ann-file does not exist: {args.val_ann_file}')
        overrides.update({
            'data.val.ann_file': args.val_ann_file,
            'data.test.ann_file': args.val_ann_file,
        })
    if args.workers_per_gpu is not None:
        if args.workers_per_gpu < 0:
            raise ValueError('--workers-per-gpu cannot be negative.')
        overrides['data.workers_per_gpu'] = args.workers_per_gpu
    if args.epochs is not None:
        overrides.update({'total_epochs': args.epochs,
                          'runner.max_epochs': args.epochs})
    if args.history_frames is not None:
        overrides.update({
            'data.train.queue_length': args.history_frames,
            'data.val.queue_length': args.history_frames,
            'data.test.queue_length': args.history_frames,
            'model.future_pred_head.history_queue_length': args.history_frames,
        })
    if args.predict_future_occ is not None:
        overrides.update({
            'data.train.future_pred_frame_num': occ_steps,
            'data.val.future_pred_frame_num': occ_steps,
            'data.test.future_pred_frame_num': occ_steps,
            'model.future_pred_frame_num': occ_steps,
            'model.test_future_frame_num': occ_steps,
        })
    if args.predict_future_traj is not None:
        overrides.update({
            'data.train.future_traj_frame_num': args.future_traj_steps,
            'data.val.future_traj_frame_num': args.future_traj_steps,
            'data.test.future_traj_frame_num': args.future_traj_steps,
            'data.train.predict_trajectory': trajectory_enabled,
            'data.val.predict_trajectory': trajectory_enabled,
            'data.test.predict_trajectory': trajectory_enabled,
            'model.turn_on_plan': trajectory_enabled,
            'model.predict_trajectory': trajectory_enabled,
            'model.plan_head.planning_steps': args.future_traj_steps,
        })
    if overrides:
        cfg.merge_from_dict(overrides)
    if args.use_fp16 == 1:
        cfg.fp16 = dict(loss_scale='dynamic')
    elif args.use_fp16 == 0 and 'fp16' in cfg:
        cfg.pop('fp16')
    if grad_accum_steps is not None and grad_accum_steps > 1:
        cfg.optimizer_config = cfg.optimizer_config.copy()
        cfg.optimizer_config.update(
            type='GradientCumulativeOptimizerHook',
            cumulative_iters=grad_accum_steps)
    if args.total_batch_size is not None:
        cfg.farmsim_total_batch_size = args.total_batch_size
        cfg.farmsim_grad_accum_steps = grad_accum_steps
    if args.use_crop_gap_refinement is not None:
        cfg.model.future_pred_head.use_crop_gap_refinement = bool(
            args.use_crop_gap_refinement)
        cfg.model.future_pred_head.crop_gap_boundary_loss_weight = (
            args.crop_gap_boundary_loss_weight)
        cfg.model.future_pred_head.crop_gap_free_loss_weight = (
            args.crop_gap_free_loss_weight)
        cfg.model.future_pred_head.crop_gap_alpha = args.crop_gap_alpha
        cfg.model.future_pred_head.crop_gap_sigma = args.crop_gap_sigma
        cfg.model.future_pred_head.crop_gap_radius = args.crop_gap_radius
    if args.use_selective_c2f is not None:
        cfg.model.future_pred_head.use_selective_c2f = bool(
            args.use_selective_c2f)
        cfg.model.future_pred_head.c2f_active_ratio = args.c2f_active_ratio
        cfg.model.future_pred_head.c2f_channels = args.c2f_channels
    if args.use_dual_hardness_refinement is not None:
        cfg.model.future_pred_head.use_dual_hardness_refinement = bool(
            args.use_dual_hardness_refinement)
        cfg.model.future_pred_head.dual_hardness_active_ratio = (
            args.dual_hardness_active_ratio)
        cfg.model.future_pred_head.dual_hardness_gap_ratio = (
            args.dual_hardness_gap_ratio)
        cfg.model.future_pred_head.dual_hardness_channels = (
            args.dual_hardness_channels)
        cfg.model.future_pred_head.dual_hardness_local_scale = (
            args.dual_hardness_local_scale)
        cfg.model.future_pred_head.dual_hardness_gap_boost = (
            args.dual_hardness_gap_boost)
        cfg.model.future_pred_head.dual_hardness_loss_weight = (
            args.dual_hardness_loss_weight)
        cfg.model.future_pred_head.dual_hardness_distill_weight = (
            args.dual_hardness_distill_weight)
        cfg.model.future_pred_head.dual_hardness_ema_decay = (
            args.dual_hardness_ema_decay)
    if args.use_gap_residual_refiner is not None:
        cfg.model.future_pred_head.use_gap_residual_refiner = bool(
            args.use_gap_residual_refiner)
        cfg.model.future_pred_head.gap_refiner_channels = args.gap_refiner_channels
        cfg.model.future_pred_head.gap_refiner_blocks = args.gap_refiner_blocks
        cfg.model.future_pred_head.gap_refiner_coarse_loss_weight = (
            args.gap_refiner_coarse_loss_weight)
        cfg.model.future_pred_head.gap_refiner_boundary_loss_weight = (
            args.gap_refiner_boundary_loss_weight)
        cfg.model.future_pred_head.gap_refiner_gap_loss_weight = (
            args.gap_refiner_gap_loss_weight)
        cfg.model.future_pred_head.gap_refiner_crop_loss_weight = (
            args.gap_refiner_crop_loss_weight)
        cfg.model.future_pred_head.gap_refiner_use_bev_feature = bool(
            args.gap_refiner_use_bev_feature)
        cfg.model.future_pred_head.gap_refiner_use_image_features = bool(
            args.gap_refiner_use_image_features)
        cfg.model.future_pred_head.gap_refiner_image_active_ratio = (
            args.gap_refiner_image_active_ratio)
        cfg.model.future_pred_head.gap_refiner_image_channels = (
            args.gap_refiner_image_channels)
        cfg.model.future_pred_head.gap_refiner_image_levels = (
            args.gap_refiner_image_levels)
        cfg.model.future_pred_head.gap_refiner_image_crop_ratio = (
            args.gap_refiner_image_crop_ratio)
    if args.use_nearfar_bev is not None:
        cfg.model.pts_bbox_head.transformer.encoder.use_nearfar_bev = bool(
            args.use_nearfar_bev)
        cfg.model.pts_bbox_head.transformer.encoder.nearfar_near_ratio = (
            args.nearfar_near_ratio)
        cfg.model.pts_bbox_head.transformer.encoder.nearfar_far_stride = (
            args.nearfar_far_stride)
    if args.disable_temporal_self_attention == 1:
        layer_cfg = cfg.model.pts_bbox_head.transformer.encoder.transformerlayers
        temporal_order = ('self_attn', 'norm', 'cross_attn', 'norm', 'ffn',
                          'norm')
        if tuple(layer_cfg.operation_order) != temporal_order:
            raise ValueError('The configured BEV encoder does not use the '
                             'standard TemporalSelfAttention layout.')
        attn_cfgs = list(layer_cfg.attn_cfgs)
        if (len(attn_cfgs) != 2 or
                attn_cfgs[0].get('type') != 'TemporalSelfAttention'):
            raise ValueError('Expected TemporalSelfAttention as the first '
                             'BEV encoder attention configuration.')
        layer_cfg.attn_cfgs = attn_cfgs[1:]
        layer_cfg.operation_order = ('cross_attn', 'norm', 'ffn', 'norm')
        cfg.farmsim_temporal_self_attention = False
    elif (args.use_gvad_attention == 1 or
          args.use_directional_decay_retention == 1):
        layer_cfg = cfg.model.pts_bbox_head.transformer.encoder.transformerlayers
        temporal_order = ('self_attn', 'norm', 'cross_attn', 'norm', 'ffn',
                          'norm')
        if tuple(layer_cfg.operation_order) != temporal_order:
            raise ValueError('The configured BEV encoder does not use the '
                             'standard TemporalSelfAttention layout.')
        attn_cfgs = list(layer_cfg.attn_cfgs)
        if (len(attn_cfgs) != 2 or
                attn_cfgs[0].get('type') != 'TemporalSelfAttention'):
            raise ValueError('Expected TemporalSelfAttention as the first '
                             'BEV encoder attention configuration.')
        mixer_cfg = dict(attn_cfgs[0])
        if args.use_gvad_attention == 1:
            mixer_cfg.update(
                type='GeometryVisibleAnchorDeformableAttention',
                num_heads=args.gvad_num_heads,
                anchor_grid_height=args.gvad_anchor_grid_height,
                anchor_grid_width=args.gvad_anchor_grid_width,
                use_visibility=bool(args.gvad_use_visibility),
                use_local_deformable=bool(args.gvad_use_local_deformable))
            if args.gvad_use_visibility and args.gvad_use_local_deformable:
                cfg.farmsim_bev_spatial_mixer = 'gvad'
            elif args.gvad_use_local_deformable:
                cfg.farmsim_bev_spatial_mixer = 'gvad_plain_anchor'
            else:
                cfg.farmsim_bev_spatial_mixer = 'gvad_anchor_only'
        else:
            mixer_cfg.update(
                type='DirectionalDecaySelectiveRetention',
                retention_radius=args.ddsr_retention_radius,
                local_dilation=args.ddsr_local_dilation)
            cfg.farmsim_bev_spatial_mixer = 'directional_decay_retention'
        attn_cfgs[0] = mixer_cfg
        layer_cfg.attn_cfgs = attn_cfgs
    if args.use_r50_image_encoder == 1:
        cfg.model.img_backbone.depth = 50
        cfg.farmsim_image_encoder = 'r50'
    if args.use_efficient_baseline:
        # Use R50 and keep only C3/C4 FPN outputs; make all image-BEV
        # attention settings consistent with the two-level representation.
        cfg.model.img_backbone.depth = 50
        neck = cfg.model.img_neck
        neck.end_level = 2
        neck.num_outs = 2

        transformer = cfg.model.pts_bbox_head.transformer
        transformer.num_feature_levels = 2
        encoder = transformer.encoder
        encoder.num_layers = 4
        deformable_attention = (
            encoder.transformerlayers.attn_cfgs[1].deformable_attention)
        deformable_attention.num_levels = 2
        deformable_attention.num_points = 4

        # MMCV 1.x's runner path in this project supports AMP FP16 (not a
        # BF16 optimizer hook), so efficient mode always enables FP16.
        cfg.fp16 = dict(loss_scale='dynamic')


def main():
    args = parse_args()

    if args.use_r50_image_encoder == 1 and args.load_from is None:
        repo_root = osp.dirname(osp.dirname(osp.abspath(__file__)))
        args.load_from = osp.join(
            repo_root, 'pretrained', 'r50_fcos3d_pretrain.pth')

    cfg = Config.fromfile(args.config)
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)
    apply_farmsim_options(cfg, args)
    # import modules from string list.
    if cfg.get('custom_imports', None):
        from mmcv.utils import import_modules_from_strings
        import_modules_from_strings(**cfg['custom_imports'])

    # import modules from plguin/xx, registry will be updated
    if hasattr(cfg, 'plugin'):
        if cfg.plugin:
            import importlib
            if hasattr(cfg, 'plugin_dir'):
                plugin_dir = cfg.plugin_dir
                _module_dir = os.path.dirname(plugin_dir)
                _module_dir = _module_dir.split('/')
                _module_path = _module_dir[0]

                for m in _module_dir[1:]:
                    _module_path = _module_path + '.' + m
                print(_module_path)
                plg_lib = importlib.import_module(_module_path)
            else:
                # import dir is the dirpath for the config file
                _module_dir = os.path.dirname(args.config)
                _module_dir = _module_dir.split('/')
                _module_path = _module_dir[0]
                for m in _module_dir[1:]:
                    _module_path = _module_path + '.' + m
                print(_module_path)
                plg_lib = importlib.import_module(_module_path)

            from projects.mmdet3d_plugin.bevformer.apis.train import custom_train_model
    # set cudnn_benchmark
    if cfg.get('cudnn_benchmark', False):
        torch.backends.cudnn.benchmark = True
    # set tf32
    if cfg.get('close_tf32', False):
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False

    # work_dir is determined in this priority: CLI > segment in file > filename
    if args.work_dir is not None:
        # update configs according to CLI args if args.work_dir is not None
        cfg.work_dir = args.work_dir
    elif cfg.get('work_dir', None) is None:
        # use config filename as default work_dir if cfg.work_dir is None
        cfg.work_dir = osp.join('./work_dirs',
                                osp.splitext(osp.basename(args.config))[0])
    if args.load_from and args.resume_from:
        raise ValueError(
            '--load-from initializes a new run while --resume-from continues '
            'an existing run; specify only one of them')
    if args.load_from:
        if not osp.isfile(args.load_from):
            raise FileNotFoundError(
                f'--load-from checkpoint does not exist: {args.load_from}')
        cfg.load_from = args.load_from
    if args.resume_from:
        if not osp.isfile(args.resume_from):
            raise FileNotFoundError(
                f'--resume-from checkpoint does not exist: {args.resume_from}')
        cfg.resume_from = args.resume_from
    if args.gpu_ids is not None:
        cfg.gpu_ids = args.gpu_ids
    else:
        cfg.gpu_ids = range(1) if args.gpus is None else range(args.gpus)
    if digit_version(TORCH_VERSION) == digit_version('1.8.1') and cfg.optimizer['type'] == 'AdamW':
        cfg.optimizer['type'] = 'AdamW2' # fix bug in Adamw
    if args.autoscale_lr:
        # apply the linear scaling rule (https://arxiv.org/abs/1706.02677)
        cfg.optimizer['lr'] = cfg.optimizer['lr'] * len(cfg.gpu_ids) / 8

    # init distributed env first, since logger depends on the dist info.
    if args.launcher == 'none':
        distributed = False
    else:
        distributed = True
        init_dist(args.launcher, **cfg.dist_params)
        # re-set gpu_ids with distributed training mode
        _, world_size = get_dist_info()
        cfg.gpu_ids = range(world_size)

    # create work_dir
    mmcv.mkdir_or_exist(osp.abspath(cfg.work_dir))
    # dump config
    cfg.dump(osp.join(cfg.work_dir, osp.basename(args.config)))
    # init the logger before other steps
    timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
    log_file = osp.join(cfg.work_dir, f'{timestamp}.log')
    # specify logger name, if we still use 'mmdet', the output info will be
    # filtered and won't be saved in the log_file
    # TODO: ugly workaround to judge whether we are training det or seg model
    if cfg.model.type in ['EncoderDecoder3D']:
        logger_name = 'mmseg'
    else:
        logger_name = 'mmdet'
    logger = get_root_logger(
        log_file=log_file, log_level=cfg.log_level, name=logger_name)

    # init the meta dict to record some important information such as
    # environment info and seed, which will be logged
    meta = dict()
    # log env info
    env_info_dict = collect_env()
    env_info = '\n'.join([(f'{k}: {v}') for k, v in env_info_dict.items()])
    dash_line = '-' * 60 + '\n'
    logger.info('Environment info:\n' + dash_line + env_info + '\n' +
                dash_line)
    meta['env_info'] = env_info
    meta['config'] = cfg.pretty_text

    # log some basic info
    logger.info(f'Distributed training: {distributed}')
    logger.info(f'Config:\n{cfg.pretty_text}')

    # set random seeds
    if args.seed is not None:
        logger.info(f'Set random seed to {args.seed}, '
                    f'deterministic: {args.deterministic}')
        set_random_seed(args.seed, deterministic=args.deterministic)
    cfg.seed = args.seed
    meta['seed'] = args.seed
    meta['exp_name'] = osp.basename(args.config)

    model = build_model(
        cfg.model,
        train_cfg=cfg.get('train_cfg'),
        test_cfg=cfg.get('test_cfg'))
    model.init_weights()

    frozen_refiner_specs = (
        ('GapRef', args.freeze_gap_refiner_base,
         cfg.model.future_pred_head.use_gap_residual_refiner,
         'future_pred_head.gap_refiner_', 'freeze_gap_refiner_base',
         '--freeze-gap-refiner-base'),
    )
    for (label, requested, enabled, parameter_prefix, model_flag,
         command_flag) in frozen_refiner_specs:
        if not requested:
            continue
        if not enabled:
            raise ValueError(
                f'{command_flag} requires its refiner to be enabled.')
        trainable_names = []
        for name, parameter in model.named_parameters():
            trainable = name.startswith(parameter_prefix)
            parameter.requires_grad_(trainable)
            if trainable:
                trainable_names.append(name)
        if not trainable_names:
            raise RuntimeError(
                f'{label} freeze requested but no {label} parameters exist.')
        setattr(model, model_flag, True)
        logger.info('%s diagnostic freeze enabled; trainable parameters: %s',
                    label, ', '.join(trainable_names))

    trainable_params = sum(p.numel() for p in model.parameters()
                           if p.requires_grad)
    logger.info('Model built: %s (%d trainable parameters)',
                type(model).__name__, trainable_params)
    datasets = [build_dataset(cfg.data.train)]
    if len(cfg.workflow) == 2:
        val_dataset = copy.deepcopy(cfg.data.val)
        # in case we use a dataset wrapper
        if 'dataset' in cfg.data.train:
            val_dataset.pipeline = cfg.data.train.dataset.pipeline
        else:
            val_dataset.pipeline = cfg.data.train.pipeline
        # set test_mode=False here in deep copied config
        # which do not affect AP/AR calculation later
        # refer to https://mmdetection3d.readthedocs.io/en/latest/tutorials/customize_runtime.html#customize-workflow  # noqa
        val_dataset.test_mode = False
        datasets.append(build_dataset(val_dataset))
    if cfg.checkpoint_config is not None:
        # save mmdet version, config file content and class names in
        # checkpoints as meta data
        cfg.checkpoint_config.meta = dict(
            mmdet_version=mmdet_version,
            mmseg_version=mmseg_version,
            mmdet3d_version=mmdet3d_version,
            config=cfg.pretty_text,
            CLASSES=datasets[0].CLASSES,
            PALETTE=datasets[0].PALETTE  # for segmentors
            if hasattr(datasets[0], 'PALETTE') else None)
    # add an attribute for visualization convenience
    model.CLASSES = datasets[0].CLASSES
    custom_train_model(
        model,
        datasets,
        cfg,
        distributed=distributed,
        validate=(not args.no_validate),
        timestamp=timestamp,
        meta=meta)


if __name__ == '__main__':
    main()
