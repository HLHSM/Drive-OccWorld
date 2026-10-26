#!/usr/bin/env bash
# Two epochs of A-D on the existing 48/16 split, including DDP and validation.
set -euo pipefail
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}" \
NUM_GPUS="${NUM_GPUS:-2}" \
BATCH_SIZE=1 TOTAL_BATCH_SIZE=4 WORKERS_PER_GPU=0 \
TRAIN_ANN_FILE=data/farmsim/splits/smoke48/train_48.json \
VAL_ANN_FILE=data/farmsim/splits/smoke48/val_16.json \
CROP_STATS_FILE=work_dirs/farmsim_front3_s_smoke48_crop_stats.json \
FARMSIM_TASKS="${FARMSIM_TASKS:-kd_control,kd_logits,kd_fpn,kd_both}" \
KD_EPOCHS=2 KD_RUN_SUFFIX=_smoke48 KD_LR_WARMUP_ITERS=2 LOG_INTERVAL=1 \
  bash tools/train_farmsim_front3_s.sh
