#!/usr/bin/env bash
# FarmSim H=0/current-occupancy: GVAD + ADHR, with/without NearFar BEV.
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

PYTHON_BIN="/home/HL/.conda/envs/dow2/bin/python"
CONFIG="projects/configs/farmsim/farmsim_occ_front3.py"
PRETRAINED_FROM="$(pwd)/pretrained/r101_dcn_fcos3d_pretrain.pth"

RUN_GVAD_ADHR="${RUN_GVAD_ADHR:-1}"
RUN_GVAD_ADHR_NEARFAR="${RUN_GVAD_ADHR_NEARFAR:-1}"

# GVAD: 32 visibility-aware global anchors plus the local deformable branch.
GVAD_NUM_HEADS=8
GVAD_ANCHOR_GRID_HEIGHT=4
GVAD_ANCHOR_GRID_WIDTH=8

# ADHR: retain the previous eight-epoch ADHR configuration.
ADHR_ACTIVE_RATIO=0.04
ADHR_GAP_RATIO=0.5
ADHR_CHANNELS=128
ADHR_LOCAL_SCALE=0.25
ADHR_GAP_BOOST=0.5
ADHR_LOSS_WEIGHT=0.5
ADHR_DISTILL_WEIGHT=0.1
ADHR_EMA_DECAY=0.99

# NearFar: dense near field, stride-2 sampled far field.
NEARFAR_NEAR_RATIO=0.6
NEARFAR_FAR_STRIDE=2

run_gvad_adhr() {
  local name="$1"
  local use_nearfar="$2"
  local work_dir="work_dirs/front3_${name}_nohis_ep${EPOCHS}_$(date +%Y%m%d_%H%M%S)"

  [[ -f "${PRETRAINED_FROM}" ]] || {
    echo "Missing pretrained checkpoint: ${PRETRAINED_FROM}" >&2
    exit 1
  }
  PYTHONPATH="$(pwd):${PYTHONPATH:-}" CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}" \
    "${PYTHON_BIN}" -m torch.distributed.run --standalone \
    --nproc_per_node="${NUM_GPUS}" tools/train.py "${CONFIG}" \
    --launcher pytorch --deterministic --work-dir "${work_dir}" \
    --load-from "${PRETRAINED_FROM}" \
    --num-gpus "${NUM_GPUS}" --data-root "${DATA_ROOT}" \
    --batch-size "${BATCH_SIZE}" --total-batch-size "${TOTAL_BATCH_SIZE}" \
    --image-width "${IMAGE_WIDTH}" --image-height "${IMAGE_HEIGHT}" \
    --use-fp16 "${USE_FP16}" --use-crop-gap-refinement 0 \
    --use-selective-c2f 0 --use-gap-residual-refiner 0 \
    --disable-temporal-self-attention 0 --use-r50-image-encoder 0 \
    --use-gvad-attention 1 --gvad-use-visibility 1 \
    --gvad-use-local-deformable 1 --gvad-num-heads "${GVAD_NUM_HEADS}" \
    --gvad-anchor-grid-height "${GVAD_ANCHOR_GRID_HEIGHT}" \
    --gvad-anchor-grid-width "${GVAD_ANCHOR_GRID_WIDTH}" \
    --use-dual-hardness-refinement 1 \
    --dual-hardness-active-ratio "${ADHR_ACTIVE_RATIO}" \
    --dual-hardness-gap-ratio "${ADHR_GAP_RATIO}" \
    --dual-hardness-channels "${ADHR_CHANNELS}" \
    --dual-hardness-local-scale "${ADHR_LOCAL_SCALE}" \
    --dual-hardness-gap-boost "${ADHR_GAP_BOOST}" \
    --dual-hardness-loss-weight "${ADHR_LOSS_WEIGHT}" \
    --dual-hardness-distill-weight "${ADHR_DISTILL_WEIGHT}" \
    --dual-hardness-ema-decay "${ADHR_EMA_DECAY}" \
    --use-nearfar-bev "${use_nearfar}" \
    --nearfar-near-ratio "${NEARFAR_NEAR_RATIO}" \
    --nearfar-far-stride "${NEARFAR_FAR_STRIDE}" \
    --history-frames 0 --predict-future-occ 0 --future-occ-steps 0 \
    --predict-future-traj 0 --future-traj-steps 6 --epochs "${EPOCHS}"
}

# # Full dense GVAD together with ADHR.
# if [[ "${RUN_GVAD_ADHR}" == "1" ]]; then
#   run_gvad_adhr "gvad_adhr" 0
# fi

# In NearFar's sparse early layers GVAD pools anchors by original dense BEV
# coordinates; after restoration its final dense layer also enables the local
# deformable path.  Thus the two modules retain their intended geometry.
if [[ "${RUN_GVAD_ADHR_NEARFAR}" == "1" ]]; then
  run_gvad_adhr "gvad_adhr_nearfar_r${NEARFAR_NEAR_RATIO}_s${NEARFAR_FAR_STRIDE}" 1
fi
