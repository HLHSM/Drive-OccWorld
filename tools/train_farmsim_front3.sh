#!/usr/bin/env bash
# First-stage FarmSim training: ADHR + NearFar BEV, H=0 current occupancy.
set -euo pipefail

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
NUM_GPUS=4
DATA_ROOT="/data/HL/SimData-Occ/SimData"
BATCH_SIZE=3
TOTAL_BATCH_SIZE=24
IMAGE_WIDTH=512
IMAGE_HEIGHT=288
EPOCHS=8
USE_FP16=1

USE_DUAL_HARDNESS_REFINEMENT=1
DUAL_HARDNESS_ACTIVE_RATIO=0.1
DUAL_HARDNESS_GAP_RATIO=0.5
DUAL_HARDNESS_CHANNELS=128
DUAL_HARDNESS_LOCAL_SCALE=0.25
DUAL_HARDNESS_GAP_BOOST=0.5
DUAL_HARDNESS_LOSS_WEIGHT=0.5
DUAL_HARDNESS_DISTILL_WEIGHT=0.1
DUAL_HARDNESS_EMA_DECAY=0.99

USE_NEARFAR_BEV=1
NEARFAR_NEAR_RATIO=0.6
NEARFAR_FAR_STRIDE=2

PYTHON_BIN="/home/HL/.conda/envs/dow2/bin/python"
CONFIG="projects/configs/farmsim/farmsim_occ_front3.py"
PRETRAINED_FROM="$(pwd)/pretrained/r101_dcn_fcos3d_pretrain.pth"
WORK_DIR="work_dirs/front3_adhr_nearfar_r${NEARFAR_NEAR_RATIO}_s${NEARFAR_FAR_STRIDE}_nohis_ep${EPOCHS}_$(date +%Y%m%d_%H%M%S)"
WORK_DIR_BASE_NEARFAR="work_dirs/front3_base_nearfar_r${NEARFAR_NEAR_RATIO}_s${NEARFAR_FAR_STRIDE}_nohis_ep${EPOCHS}_$(date +%Y%m%d_%H%M%S)"

PYTHONPATH="$(pwd):${PYTHONPATH:-}" CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}" \
  "${PYTHON_BIN}" -m torch.distributed.run --standalone \
  --nproc_per_node="${NUM_GPUS}" tools/train.py "${CONFIG}" \
  --launcher pytorch --deterministic --work-dir "${WORK_DIR}" \
  --load-from "${PRETRAINED_FROM}" \
  --num-gpus "${NUM_GPUS}" --data-root "${DATA_ROOT}" \
  --batch-size "${BATCH_SIZE}" --total-batch-size "${TOTAL_BATCH_SIZE}" \
  --image-width "${IMAGE_WIDTH}" --image-height "${IMAGE_HEIGHT}" \
  --use-fp16 "${USE_FP16}" --use-tghd 0 --use-acfs-bev 0 \
  --use-row-topology 0 --use-crop-gap-refinement 0 \
  --use-selective-c2f 0 --use-fixed-group-decoder 0 \
  --use-gap-residual-refiner 0 \
  --use-dual-hardness-refinement "${USE_DUAL_HARDNESS_REFINEMENT}" \
  --dual-hardness-active-ratio "${DUAL_HARDNESS_ACTIVE_RATIO}" \
  --dual-hardness-gap-ratio "${DUAL_HARDNESS_GAP_RATIO}" \
  --dual-hardness-channels "${DUAL_HARDNESS_CHANNELS}" \
  --dual-hardness-local-scale "${DUAL_HARDNESS_LOCAL_SCALE}" \
  --dual-hardness-gap-boost "${DUAL_HARDNESS_GAP_BOOST}" \
  --dual-hardness-loss-weight "${DUAL_HARDNESS_LOSS_WEIGHT}" \
  --dual-hardness-distill-weight "${DUAL_HARDNESS_DISTILL_WEIGHT}" \
  --dual-hardness-ema-decay "${DUAL_HARDNESS_EMA_DECAY}" \
  --use-nearfar-bev "${USE_NEARFAR_BEV}" \
  --nearfar-near-ratio "${NEARFAR_NEAR_RATIO}" \
  --nearfar-far-stride "${NEARFAR_FAR_STRIDE}" \
  --history-frames 0 --predict-future-occ 0 --future-occ-steps 0 \
  --predict-future-traj 0 --future-traj-steps 6 --epochs "${EPOCHS}"

# Matched NearFar-only baseline: all settings above are held fixed except ADHR.
PYTHONPATH="$(pwd):${PYTHONPATH:-}" CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}" \
  "${PYTHON_BIN}" -m torch.distributed.run --standalone \
  --nproc_per_node="${NUM_GPUS}" tools/train.py "${CONFIG}" \
  --launcher pytorch --deterministic --work-dir "${WORK_DIR_BASE_NEARFAR}" \
  --load-from "${PRETRAINED_FROM}" \
  --num-gpus "${NUM_GPUS}" --data-root "${DATA_ROOT}" \
  --batch-size "${BATCH_SIZE}" --total-batch-size "${TOTAL_BATCH_SIZE}" \
  --image-width "${IMAGE_WIDTH}" --image-height "${IMAGE_HEIGHT}" \
  --use-fp16 "${USE_FP16}" --use-tghd 0 --use-acfs-bev 0 \
  --use-row-topology 0 --use-crop-gap-refinement 0 \
  --use-selective-c2f 0 --use-fixed-group-decoder 0 \
  --use-gap-residual-refiner 0 --use-dual-hardness-refinement 0 \
  --use-nearfar-bev "${USE_NEARFAR_BEV}" \
  --nearfar-near-ratio "${NEARFAR_NEAR_RATIO}" \
  --nearfar-far-stride "${NEARFAR_FAR_STRIDE}" \
  --history-frames 0 --predict-future-occ 0 --future-occ-steps 0 \
  --predict-future-traj 0 --future-traj-steps 6 --epochs "${EPOCHS}"
