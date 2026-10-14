"""Server-only checkpoint and dataset adapter (not required on Orin runtime)."""
import importlib
import json
from pathlib import Path
import numpy as np
import sitecustomize  # noqa: F401
import torch
from mmcv import Config
from mmcv.parallel import DataContainer
from mmcv.runner import load_checkpoint
from mmdet3d.datasets import build_dataset
from mmdet3d.models import build_model

DEFAULT_RUN = 'work_dirs/front3_s_base_agri_amoe96_cropsample0.1_min500_nohis_b24_ep15_20261012_214738'
DEFAULT_CONFIG = DEFAULT_RUN + '/farmsim_occ_front3_s.py'
DEFAULT_CHECKPOINT = DEFAULT_RUN + '/epoch_15.pth'

def configuration(path):
    cfg = Config.fromfile(path)
    if cfg.get('plugin', False):
        importlib.import_module(cfg.plugin_dir.rstrip('/').replace('/', '.'))
    return cfg

def load_model(config, checkpoint):
    cfg = configuration(config)
    cfg.model.train_cfg = None
    model = build_model(cfg.model, test_cfg=cfg.get('test_cfg'))
    load_checkpoint(model, checkpoint, map_location='cpu')
    # Deployment takes over precision explicitly; prevent MMCV decorators
    # from silently changing a FP32 reference run into mixed precision.
    for module in model.modules():
        if hasattr(module, 'fp16_enabled'):
            module.fp16_enabled = False
    if model.future_pred_head.history_queue_length or model.test_future_frame_num or model.turn_on_plan:
        raise ValueError('Deployment currently requires current-only, no-history, no-planning config')
    if not model.future_pred_head.use_agri_amoe_decoder:
        raise ValueError('Expected an Agri-AMoE checkpoint/config')
    unsupported=['use_crop_gap_refinement','use_selective_c2f','use_gap_residual_refiner',
                 'use_occupancy_2x_refiner','use_dual_hardness_refinement']
    enabled=[name for name in unsupported if getattr(model.future_pred_head,name,False)]
    if enabled: raise ValueError(f'Unsupported extra inference/refinement branches: {enabled}')
    return cfg, model.cuda().eval()

def unwrap(value):
    if isinstance(value, DataContainer):
        return unwrap(value.data)
    if isinstance(value, (list, tuple)):
        if len(value) == 1:
            return unwrap(value[0])
        return [unwrap(item) for item in value]
    return value

def samples(config, count=0, split='test', sampling='leading'):
    from projects.mmdet3d_plugin.datasets.builder import build_dataloader
    cfg = configuration(config)
    dataset_cfg = cfg.data[split].copy()
    if split == 'train':
        # Calibrate using training frames, but deterministic test transforms.
        test_cfg = cfg.data.test
        dataset_cfg.pipeline = test_cfg.pipeline
        dataset_cfg.test_mode = True
        dataset_cfg.queue_length = 0
        dataset_cfg.future_queue_length = 0
        dataset_cfg.use_crop_positive_sampling = False
    if count and split != 'train' and sampling == 'leading':
        dataset_cfg.max_samples = count
    dataset_cfg.samples_per_gpu = 1
    dataset = build_dataset(dataset_cfg)
    if count and split == 'train':
        # Uniform seeded selection across the complete training manifest.
        # Modify the sample list, leaving the dataset's metadata interface intact.
        chosen = np.random.default_rng(0).choice(len(dataset.samples), min(count,len(dataset.samples)), replace=False)
        dataset.samples = [dataset.samples[i] for i in chosen]
        dataset.flag = np.zeros(len(dataset.samples), dtype=np.uint8)
    elif count and sampling == 'uniform':
        chosen = np.linspace(0,len(dataset.samples)-1,min(count,len(dataset.samples)),dtype=np.int64)
        dataset.samples = [dataset.samples[i] for i in chosen]
        dataset.flag = np.zeros(len(dataset.samples), dtype=np.uint8)
    loader = build_dataloader(dataset, 1, 0, dist=False, shuffle=False)
    for index, data in enumerate(loader):
        image = unwrap(data['img'])
        while image.ndim > 4:
            if image.shape[0] != 1:
                raise ValueError(f'Expected batch/queue=1, got {image.shape}')
            image = image[0]
        metadata = unwrap(data['img_metas'])
        if isinstance(metadata, dict) and 'lidar2img' not in metadata:
            metadata = metadata[max(metadata)]
        if isinstance(metadata, list):
            metadata = metadata[-1]
        target = unwrap(data['segmentation'])
        while target.ndim > 3:
            target = target[0]
        yield image.contiguous(), metadata, target.long(), data
        if count and index + 1 >= count:
            break

def write_json(path, result):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=2, ensure_ascii=False)+'\n', encoding='utf-8')

def labels(logits, height, width):
    # Preserve evaluate_occ's token -> X/Y convention, including transpose.
    return logits.permute(0, 3, 1, 2).reshape(-1, logits.shape[-1], width, height,
                                            logits.shape[-2]).transpose(2, 3).argmax(1)

def metrics(hist, names):
    diagonal = np.diag(hist).astype(float)
    union = hist.sum(0) + hist.sum(1) - diagonal
    iou = np.divide(diagonal, union, out=np.zeros_like(diagonal), where=union>0)
    # Match core/evaluation/eval_hooks.py: all present classes, including free.
    present = union > 0
    result = {'miou':float(iou[present].mean()),
              'miou_all':float(iou.mean()),
              'class_iou':dict(zip(names, map(float, iou))), 'confusion_matrix':hist.tolist()}
    if 'crop' in names:
        k = names.index('crop')
        result['crop_precision'] = float(diagonal[k]/max(hist[:,k].sum(),1))
        result['crop_recall'] = float(diagonal[k]/max(hist[k,:].sum(),1))
    return result
