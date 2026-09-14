#!/usr/bin/env bash
# FarmSim front3: default Agri-AMoE + late zero-start GVADV2 + NearFar.
set -euo pipefail

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
NUM_GPUS="${NUM_GPUS:-4}"
DATA_ROOT="${DATA_ROOT:-/data/HL/SimData-Occ/SimData}"
BATCH_SIZE="${BATCH_SIZE:-2}"
TOTAL_BATCH_SIZE="${TOTAL_BATCH_SIZE:-24}"
IMAGE_WIDTH="${IMAGE_WIDTH:-512}"
IMAGE_HEIGHT="${IMAGE_HEIGHT:-288}"
EPOCHS="${EPOCHS:-8}"
USE_FP16="${USE_FP16:-1}"

PYTHON_BIN="${PYTHON_BIN:-/home/HL/.conda/envs/dow2/bin/python}"
CONFIG="${CONFIG:-projects/configs/farmsim/farmsim_occ_front3.py}"
PRETRAINED_FROM="${PRETRAINED_FROM:-$(pwd)/pretrained/r101_dcn_fcos3d_pretrain.pth}"

# GVADV2: only the final dense layers use local deformable attention plus a
# zero-initialized geometry-visible anchor residual.
GVAD_NUM_HEADS=8
GVAD_ANCHOR_GRID_HEIGHT=4
GVAD_ANCHOR_GRID_WIDTH=8
GVADV2_NUM_LAYERS=2

# Default Agri-AMoE settings selected as the fixed reference after ablations.
AGRI_AMOE_CHANNELS=96
AGRI_AMOE_USE_GRADIENT_ENERGY=1
AGRI_AMOE_USE_SALIENCY=1
AGRI_AMOE_GATE_TEMPERATURE=1.0

# NearFar: dense near field, stride-2 sampled far field.
NEARFAR_NEAR_RATIO=0.6
NEARFAR_FAR_STRIDE=2

work_dir="work_dirs/front3_agri_amoe_gvadv2_tail${GVADV2_NUM_LAYERS}_nearfar_r${NEARFAR_NEAR_RATIO}_s${NEARFAR_FAR_STRIDE}_ep${EPOCHS}_$(date +%Y%m%d_%H%M%S)"

[[ -f "${PRETRAINED_FROM}" ]] || {
  echo "Missing pretrained checkpoint: ${PRETRAINED_FROM}" >&2
  exit 1
}

# PYTHONPATH="$(pwd):${PYTHONPATH:-}" CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}" \
#   "${PYTHON_BIN}" -m torch.distributed.run --standalone \
#   --nproc_per_node="${NUM_GPUS}" tools/train.py "${CONFIG}" \
#   --launcher pytorch --deterministic --work-dir "${work_dir}" \
#   --load-from "${PRETRAINED_FROM}" \
#   --num-gpus "${NUM_GPUS}" --data-root "${DATA_ROOT}" \
#   --batch-size "${BATCH_SIZE}" --total-batch-size "${TOTAL_BATCH_SIZE}" \
#   --image-width "${IMAGE_WIDTH}" --image-height "${IMAGE_HEIGHT}" \
#   --use-fp16 "${USE_FP16}" \
#   --use-crop-gap-refinement 0 --use-selective-c2f 0 \
#   --use-gap-residual-refiner 0 --use-dual-hardness-refinement 0 \
#   --use-gvad-attention 0 --use-gvadv2-attention 1 \
#   --gvadv2-num-layers "${GVADV2_NUM_LAYERS}" \
#   --gvad-use-visibility 1 --gvad-use-local-deformable 1 \
#   --gvad-num-heads "${GVAD_NUM_HEADS}" \
#   --gvad-anchor-grid-height "${GVAD_ANCHOR_GRID_HEIGHT}" \
#   --gvad-anchor-grid-width "${GVAD_ANCHOR_GRID_WIDTH}" \
#   --use-agri-amoe-decoder 1 \
#   --agri-amoe-channels "${AGRI_AMOE_CHANNELS}" \
#   --agri-amoe-use-gradient-energy "${AGRI_AMOE_USE_GRADIENT_ENERGY}" \
#   --agri-amoe-use-saliency "${AGRI_AMOE_USE_SALIENCY}" \
#   --agri-amoe-gate-temperature "${AGRI_AMOE_GATE_TEMPERATURE}" \
#   --use-nearfar-bev 1 --nearfar-near-ratio "${NEARFAR_NEAR_RATIO}" \
#   --nearfar-far-stride "${NEARFAR_FAR_STRIDE}" \
#   --history-frames 0 --predict-future-occ 0 --future-occ-steps 0 \
#   --predict-future-traj 0 --future-traj-steps 6 --epochs "${EPOCHS}"
