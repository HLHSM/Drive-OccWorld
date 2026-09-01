#!/usr/bin/env bash

# 48 training samples / 16 validation samples, used only for end-to-end checks.
# Two-card IR-WM prediction-task smoke test: CUDA device IDs 0 and 1.
CUDA_VISIBLE_DEVICES="0,1"
NUM_GPUS=2
DATA_ROOT="/data/HL/SimData-Occ/SimData"
TRAIN_ANN_FILE="data/farmsim/splits/smoke48/train_48.json"
VAL_ANN_FILE="data/farmsim/splits/smoke48/val_16.json"

BATCH_SIZE=1
# Desired effective batch across all GPUs and accumulation steps.
TOTAL_BATCH_SIZE=2
WORKERS_PER_GPU=0
IMAGE_WIDTH=512
IMAGE_HEIGHT=288
USE_FP16=1
USE_TGHD=0
USE_ACFS_BEV=0
ACFS_ACTIVE_RATIO=0.5
# Set USE_GAP_RESIDUAL_REFINER=1 and GAP_REFINER_USE_IMAGE_FEATURES=1 to
# smoke-test the new image-guided GapRef path.
USE_ROW_TOPOLOGY="${USE_ROW_TOPOLOGY:-0}"
ROW_TOPOLOGY_LOSS_WEIGHT="${ROW_TOPOLOGY_LOSS_WEIGHT:-0.1}"
USE_CROP_GAP_REFINEMENT="${USE_CROP_GAP_REFINEMENT:-0}"
CROP_GAP_BOUNDARY_LOSS_WEIGHT="${CROP_GAP_BOUNDARY_LOSS_WEIGHT:-0.5}"
CROP_GAP_FREE_LOSS_WEIGHT="${CROP_GAP_FREE_LOSS_WEIGHT:-0.25}"
CROP_GAP_ALPHA="${CROP_GAP_ALPHA:-3.0}"
CROP_GAP_SIGMA="${CROP_GAP_SIGMA:-1.5}"
CROP_GAP_RADIUS="${CROP_GAP_RADIUS:-4}"
USE_SELECTIVE_C2F="${USE_SELECTIVE_C2F:-0}"
C2F_ACTIVE_RATIO="${C2F_ACTIVE_RATIO:-0.25}"
C2F_CHANNELS="${C2F_CHANNELS:-128}"
USE_DUAL_HARDNESS_REFINEMENT="${USE_DUAL_HARDNESS_REFINEMENT:-0}"
DUAL_HARDNESS_ACTIVE_RATIO="${DUAL_HARDNESS_ACTIVE_RATIO:-0.04}"
DUAL_HARDNESS_GAP_RATIO="${DUAL_HARDNESS_GAP_RATIO:-0.5}"
DUAL_HARDNESS_CHANNELS="${DUAL_HARDNESS_CHANNELS:-128}"
DUAL_HARDNESS_LOCAL_SCALE="${DUAL_HARDNESS_LOCAL_SCALE:-0.25}"
DUAL_HARDNESS_GAP_BOOST="${DUAL_HARDNESS_GAP_BOOST:-0.5}"
DUAL_HARDNESS_LOSS_WEIGHT="${DUAL_HARDNESS_LOSS_WEIGHT:-0.5}"
DUAL_HARDNESS_DISTILL_WEIGHT="${DUAL_HARDNESS_DISTILL_WEIGHT:-0.1}"
DUAL_HARDNESS_EMA_DECAY="${DUAL_HARDNESS_EMA_DECAY:-0.99}"
USE_FIXED_GROUP_DECODER="${USE_FIXED_GROUP_DECODER:-0}"
GROUP_DECODER_LOSS_WEIGHT="${GROUP_DECODER_LOSS_WEIGHT:-0.3}"
GROUP_DECODER_PRIOR_SCALE="${GROUP_DECODER_PRIOR_SCALE:-1.0}"
USE_GAP_RESIDUAL_REFINER="${USE_GAP_RESIDUAL_REFINER:-0}"
GAP_REFINER_CHANNELS="${GAP_REFINER_CHANNELS:-24}"
GAP_REFINER_BLOCKS="${GAP_REFINER_BLOCKS:-3}"
GAP_REFINER_COARSE_LOSS_WEIGHT="${GAP_REFINER_COARSE_LOSS_WEIGHT:-0.15}"
GAP_REFINER_BOUNDARY_LOSS_WEIGHT="${GAP_REFINER_BOUNDARY_LOSS_WEIGHT:-0.25}"
GAP_REFINER_GAP_LOSS_WEIGHT="${GAP_REFINER_GAP_LOSS_WEIGHT:-0.5}"
GAP_REFINER_CROP_LOSS_WEIGHT="${GAP_REFINER_CROP_LOSS_WEIGHT:-0.5}"
GAP_REFINER_USE_BEV_FEATURE="${GAP_REFINER_USE_BEV_FEATURE:-1}"
GAP_REFINER_USE_IMAGE_FEATURES="${GAP_REFINER_USE_IMAGE_FEATURES:-0}"
GAP_REFINER_IMAGE_ACTIVE_RATIO="${GAP_REFINER_IMAGE_ACTIVE_RATIO:-0.08}"
GAP_REFINER_IMAGE_CHANNELS="${GAP_REFINER_IMAGE_CHANNELS:-24}"
GAP_REFINER_IMAGE_LEVELS="${GAP_REFINER_IMAGE_LEVELS:-2}"
GAP_REFINER_IMAGE_CROP_RATIO="${GAP_REFINER_IMAGE_CROP_RATIO:-0.5}"
USE_NEARFAR_BEV="${USE_NEARFAR_BEV:-0}"
NEARFAR_NEAR_RATIO="${NEARFAR_NEAR_RATIO:-0.6}"
NEARFAR_FAR_STRIDE="${NEARFAR_FAR_STRIDE:-2}"
USE_EFFICIENT_BASELINE="${USE_EFFICIENT_BASELINE:-0}"
EPOCHS="${EPOCHS:-1}"
HISTORY_FRAMES="${HISTORY_FRAMES:-2}"
PREDICT_FUTURE_OCC="${PREDICT_FUTURE_OCC:-1}"
FUTURE_OCC_STEPS="${FUTURE_OCC_STEPS:-5}"
PREDICT_FUTURE_TRAJ="${PREDICT_FUTURE_TRAJ:-1}"
FUTURE_TRAJ_STEPS="${FUTURE_TRAJ_STEPS:-6}"

PYTHON_BIN="/home/HL/.conda/envs/dow2/bin/python"
WORK_DIR="${WORK_DIR:-work_dirs/front3_irwm_smoke48_$(date +%Y%m%d_%H%M%S)}"
PRETRAINED_FROM="$(pwd)/pretrained/r101_dcn_fcos3d_pretrain.pth"
CONFIG="projects/configs/farmsim/farmsim_occ_front3.py"

"${PYTHON_BIN}" tools/create_farmsim_smoke_split.py
PYTHONPATH="$(pwd):${PYTHONPATH:-}" CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}" \
  "${PYTHON_BIN}" -m torch.distributed.run --standalone \
  --nproc_per_node="${NUM_GPUS}" tools/train.py "${CONFIG}" \
  --launcher pytorch --deterministic --work-dir "${WORK_DIR}" \
  --load-from "${PRETRAINED_FROM}" \
  --num-gpus "${NUM_GPUS}" --data-root "${DATA_ROOT}" \
  --train-ann-file "${TRAIN_ANN_FILE}" --val-ann-file "${VAL_ANN_FILE}" \
  --batch-size "${BATCH_SIZE}" --workers-per-gpu "${WORKERS_PER_GPU}" \
  --image-width "${IMAGE_WIDTH}" --image-height "${IMAGE_HEIGHT}" \
  --use-fp16 "${USE_FP16}" --use-tghd "${USE_TGHD}" \
  --use-acfs-bev "${USE_ACFS_BEV}" --acfs-active-ratio "${ACFS_ACTIVE_RATIO}" \
  --use-row-topology "${USE_ROW_TOPOLOGY}" --row-topology-loss-weight "${ROW_TOPOLOGY_LOSS_WEIGHT}" \
  --use-crop-gap-refinement "${USE_CROP_GAP_REFINEMENT}" \
  --crop-gap-boundary-loss-weight "${CROP_GAP_BOUNDARY_LOSS_WEIGHT}" \
  --crop-gap-free-loss-weight "${CROP_GAP_FREE_LOSS_WEIGHT}" \
  --crop-gap-alpha "${CROP_GAP_ALPHA}" --crop-gap-sigma "${CROP_GAP_SIGMA}" \
  --crop-gap-radius "${CROP_GAP_RADIUS}" \
  --use-selective-c2f "${USE_SELECTIVE_C2F}" \
  --c2f-active-ratio "${C2F_ACTIVE_RATIO}" --c2f-channels "${C2F_CHANNELS}" \
  --use-dual-hardness-refinement "${USE_DUAL_HARDNESS_REFINEMENT}" \
  --dual-hardness-active-ratio "${DUAL_HARDNESS_ACTIVE_RATIO}" \
  --dual-hardness-gap-ratio "${DUAL_HARDNESS_GAP_RATIO}" \
  --dual-hardness-channels "${DUAL_HARDNESS_CHANNELS}" \
  --dual-hardness-local-scale "${DUAL_HARDNESS_LOCAL_SCALE}" \
  --dual-hardness-gap-boost "${DUAL_HARDNESS_GAP_BOOST}" \
  --dual-hardness-loss-weight "${DUAL_HARDNESS_LOSS_WEIGHT}" \
  --dual-hardness-distill-weight "${DUAL_HARDNESS_DISTILL_WEIGHT}" \
  --dual-hardness-ema-decay "${DUAL_HARDNESS_EMA_DECAY}" \
  --use-fixed-group-decoder "${USE_FIXED_GROUP_DECODER}" \
  --group-decoder-loss-weight "${GROUP_DECODER_LOSS_WEIGHT}" \
  --group-decoder-prior-scale "${GROUP_DECODER_PRIOR_SCALE}" \
  --use-gap-residual-refiner "${USE_GAP_RESIDUAL_REFINER}" \
  --gap-refiner-channels "${GAP_REFINER_CHANNELS}" \
  --gap-refiner-blocks "${GAP_REFINER_BLOCKS}" \
  --gap-refiner-coarse-loss-weight "${GAP_REFINER_COARSE_LOSS_WEIGHT}" \
  --gap-refiner-boundary-loss-weight "${GAP_REFINER_BOUNDARY_LOSS_WEIGHT}" \
  --gap-refiner-gap-loss-weight "${GAP_REFINER_GAP_LOSS_WEIGHT}" \
  --gap-refiner-crop-loss-weight "${GAP_REFINER_CROP_LOSS_WEIGHT}" \
  --gap-refiner-use-bev-feature "${GAP_REFINER_USE_BEV_FEATURE}" \
  --gap-refiner-use-image-features "${GAP_REFINER_USE_IMAGE_FEATURES}" \
  --gap-refiner-image-active-ratio "${GAP_REFINER_IMAGE_ACTIVE_RATIO}" \
  --gap-refiner-image-channels "${GAP_REFINER_IMAGE_CHANNELS}" \
  --gap-refiner-image-levels "${GAP_REFINER_IMAGE_LEVELS}" \
  --gap-refiner-image-crop-ratio "${GAP_REFINER_IMAGE_CROP_RATIO}" \
  --use-nearfar-bev "${USE_NEARFAR_BEV}" --nearfar-near-ratio "${NEARFAR_NEAR_RATIO}" --nearfar-far-stride "${NEARFAR_FAR_STRIDE}" \
  --use-efficient-baseline "${USE_EFFICIENT_BASELINE}" \
  --total-batch-size "${TOTAL_BATCH_SIZE}" --epochs "${EPOCHS}" \
  --history-frames "${HISTORY_FRAMES}" \
  --predict-future-occ "${PREDICT_FUTURE_OCC}" \
  --future-occ-steps "${FUTURE_OCC_STEPS}" \
  --predict-future-traj "${PREDICT_FUTURE_TRAJ}" \
  --future-traj-steps "${FUTURE_TRAJ_STEPS}" \
  --cfg-options model.future_pred_head.use_plan_traj=True
