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
    farmsim.add_argument('--use-tghd', type=int, choices=(0, 1),
                         help='enable the Terrain-Normalized Geometry-Semantic Height Decoder')
    farmsim.add_argument('--use-acfs-bev', type=int, choices=(0, 1),
                         help='enable Agriculture-aware Coarse-to-Fine Sparse BEV')
    farmsim.add_argument('--acfs-active-ratio', type=float, default=0.5,
                         help='fraction of BEV queries receiving full attention')
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
        if value is not None and value < 1:
            raise ValueError(f'{name} must be positive, got {value}.')
    if args.future_occ_steps < 0 or args.future_traj_steps < 0:
        raise ValueError('future occupancy and trajectory step counts cannot be negative.')
    if not 0.0 < args.acfs_active_ratio <= 1.0:
        raise ValueError('--acfs-active-ratio must be in (0, 1].')
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
        args.predict_future_traj, args.use_fp16, args.use_tghd,
        args.use_acfs_bev, args.use_efficient_baseline,
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
    if args.use_tghd is not None:
        cfg.model.future_pred_head.use_tghd = bool(args.use_tghd)
        cfg.data.train.return_ground_height = bool(args.use_tghd)
    if args.use_acfs_bev is not None:
        cfg.model.pts_bbox_head.transformer.encoder.use_acfs_bev = bool(
            args.use_acfs_bev)
        cfg.model.pts_bbox_head.transformer.encoder.acfs_active_ratio = (
            args.acfs_active_ratio)
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
