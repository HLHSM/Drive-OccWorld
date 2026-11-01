#!/usr/bin/env bash
# 前视三相机，前方 15×15 m，75×75×25 网格；两组各训练 15 轮。
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"
export PYTHONPATH="${PWD}:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1
PYTHON_BIN="${PYTHON_BIN:-/home/HL/.conda/envs/dow2/bin/python}"
NUM_GPUS="${NUM_GPUS:-2}"

SAMPLE_STRIDE="${SAMPLE_STRIDE:-2}"  # 训练、验证集每隔几帧取一帧。

DATA_ROOT=/data/HL/SimData-panoptic/SimData-panoptic
SPLIT_DIR=data/simdata_panoptic/splits
CROP_STATS_FILE="${CROP_STATS_FILE:-work_dirs/simdata_panoptic_front3_s_train_crop_stats.json}"
WORK_ROOT="${WORK_ROOT:-work_dirs}"
RUN_ID=$(date +%Y%m%d_%H%M%S)

# 复用已有划分，首次运行时生成训练集的 CropSample 统计。
"${PYTHON_BIN}" tools/create_simdata_panoptic_split.py \
  --data-root "${DATA_ROOT}" --output-dir "${SPLIT_DIR}" \
  --seed 20261028 --reuse-if-current
"${PYTHON_BIN}" tools/build_farmsim_crop_stats.py \
  --manifest "${SPLIT_DIR}/train.json" --data-root "${DATA_ROOT}" \
  --target-point-cloud-range 0 -7.5 -2 15 7.5 3 --output "${CROP_STATS_FILE}"

TRAIN_ARGS=(
  projects/configs/farmsim/farmsim_occ_panoptic_front3_s.py
  --launcher pytorch --deterministic --seed 0
  --load-from pretrained/r101_dcn_fcos3d_pretrain.pth
  --num-gpus "${NUM_GPUS}" --data-root "${DATA_ROOT}"
  --train-ann-file "${SPLIT_DIR}/train.json"
  --workers-per-gpu 4 --batch-size 4 --total-batch-size 24
  --image-width 512 --image-height 288 --camera-geometry grid_world
  --use-fp16 1 --epochs 15 --history-frames 0
  --predict-future-occ 0 --predict-future-traj 0
  --disable-temporal-self-attention 0 --use-r50-image-encoder 0
  --use-gvadv2-attention 0 --use-directional-decay-retention 0 --use-nearfar-bev 0
  --use-crop-gap-refinement 0 --use-selective-c2f 0
  --use-gap-residual-refiner 0 --use-dual-hardness-refinement 0
  --use-agri-amoe-decoder 1 --agri-amoe-channels 96
  --agri-amoe-use-gradient-energy 1 --agri-amoe-use-saliency 1
  --agri-amoe-gate-temperature 1.0
  --gvad-use-visibility 1 --gvad-use-local-deformable 1
  --gvad-num-heads 8 --gvad-anchor-grid-height 4 --gvad-anchor-grid-width 8
  --use-crop-positive-sampling 1 --crop-stats-file "${CROP_STATS_FILE}"
  --crop-oversample-ratio 0.1 --crop-oversample-min-voxels 500
  --cfg-options log_config.interval=20 evaluation.interval=1
  "data.train.sample_stride=${SAMPLE_STRIDE}"
  "data.val.sample_stride=${SAMPLE_STRIDE}"
  "data.val.ann_file=${SPLIT_DIR}/val.json"
  "data.test.ann_file=${SPLIT_DIR}/test.json"
)

BASE_WORK_DIR="${WORK_ROOT}/panodata_front3_s_base_amoe_cropsample_str${SAMPLE_STRIDE}_${RUN_ID}"
GVAD_WORK_DIR="${WORK_ROOT}/panodata_front3_s_base_gvad_amoe_cropsample_str${SAMPLE_STRIDE}_${RUN_ID}"
mkdir -p "${BASE_WORK_DIR}" "${GVAD_WORK_DIR}"

# Base + Agri-AMoE + CropSample
"${PYTHON_BIN}" -m torch.distributed.run --standalone \
  --nproc_per_node="${NUM_GPUS}" --log-dir "${BASE_WORK_DIR}/torchrun" --tee 3 \
  tools/train.py "${TRAIN_ARGS[@]}" \
  --use-gvad-attention 0 --work-dir "${BASE_WORK_DIR}" \
  2>&1 | tee "${BASE_WORK_DIR}/launcher.log"

# Base + GVAD + Agri-AMoE + CropSample
"${PYTHON_BIN}" -m torch.distributed.run --standalone \
  --nproc_per_node="${NUM_GPUS}" --log-dir "${GVAD_WORK_DIR}/torchrun" --tee 3 \
  tools/train.py "${TRAIN_ARGS[@]}" \
  --use-gvad-attention 1 --work-dir "${GVAD_WORK_DIR}" \
  2>&1 | tee "${GVAD_WORK_DIR}/launcher.log"
