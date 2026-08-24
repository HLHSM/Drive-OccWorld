#!/usr/bin/env bash

# 48 training samples / 16 validation samples, used only for end-to-end checks.
# Two-card ACFS smoke test: CUDA device IDs 2 and 3.
CUDA_VISIBLE_DEVICES="2,3"
NUM_GPUS=2
DATA_ROOT="/data/HL/SimData-Occ/SimData"
TRAIN_ANN_FILE="data/farmsim/splits/smoke48/train_48.json"
VAL_ANN_FILE="data/farmsim/splits/smoke48/val_16.json"

BATCH_SIZE=1
# Desired effective batch across all GPUs and accumulation steps.
TOTAL_BATCH_SIZE=2
WORKERS_PER_GPU=2
IMAGE_WIDTH=512
IMAGE_HEIGHT=288
USE_FP16=1
USE_TGHD=0
USE_ACFS_BEV=1
ACFS_ACTIVE_RATIO=0.5
USE_EFFICIENT_BASELINE=0
EPOCHS=1
HISTORY_FRAMES=2
PREDICT_FUTURE_OCC=0
FUTURE_OCC_STEPS=0
PREDICT_FUTURE_TRAJ=0
FUTURE_TRAJ_STEPS=6

PYTHON_BIN="/home/HL/.conda/envs/dow2/bin/python"
WORK_DIR="work_dirs/front3_smoke48_$(date +%Y%m%d_%H%M%S)"
CONFIG="projects/configs/farmsim/farmsim_occ_front3.py"

"${PYTHON_BIN}" tools/create_farmsim_smoke_split.py
PYTHONPATH="$(pwd):${PYTHONPATH:-}" CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}" \
  "${PYTHON_BIN}" -m torch.distributed.run --standalone \
  --nproc_per_node="${NUM_GPUS}" tools/train.py "${CONFIG}" \
  --launcher pytorch --deterministic --work-dir "${WORK_DIR}" \
  --num-gpus "${NUM_GPUS}" --data-root "${DATA_ROOT}" \
  --train-ann-file "${TRAIN_ANN_FILE}" --val-ann-file "${VAL_ANN_FILE}" \
  --batch-size "${BATCH_SIZE}" --workers-per-gpu "${WORKERS_PER_GPU}" \
  --image-width "${IMAGE_WIDTH}" --image-height "${IMAGE_HEIGHT}" \
  --use-fp16 "${USE_FP16}" --use-tghd "${USE_TGHD}" \
  --use-acfs-bev "${USE_ACFS_BEV}" --acfs-active-ratio "${ACFS_ACTIVE_RATIO}" \
  --use-efficient-baseline "${USE_EFFICIENT_BASELINE}" \
  --total-batch-size "${TOTAL_BATCH_SIZE}" --epochs "${EPOCHS}" \
  --history-frames "${HISTORY_FRAMES}" \
  --predict-future-occ "${PREDICT_FUTURE_OCC}" \
  --future-occ-steps "${FUTURE_OCC_STEPS}" \
  --predict-future-traj "${PREDICT_FUTURE_TRAJ}" \
  --future-traj-steps "${FUTURE_TRAJ_STEPS}"
