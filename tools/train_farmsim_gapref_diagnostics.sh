#!/usr/bin/env bash
# Second-stage image-guided GapRef training on two first-stage bases.
# Both runs freeze every non-GapRef parameter and use only the primary
# occupancy loss; all previous GapRef auxiliary-loss weights are zero.
set -euo pipefail

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
NUM_GPUS="${NUM_GPUS:-4}"
DATA_ROOT="${DATA_ROOT:-/data/HL/SimData-Occ/SimData}"
PYTHON_BIN="${PYTHON_BIN:-/home/HL/.conda/envs/dow2/bin/python}"
BASE_CONFIG="${BASE_CONFIG:-projects/configs/farmsim/farmsim_occ_front3.py}"
VAL_ANN_FILE="${VAL_ANN_FILE:-data/farmsim/splits/val.json}"

ADHR_BASE_CHECKPOINT="${ADHR_BASE_CHECKPOINT:-work_dirs/front3_base_adhr_nohis_ep8_20260829_085702/epoch_8.pth}"
NEARFAR_BASE_CHECKPOINT="${NEARFAR_BASE_CHECKPOINT:-work_dirs/front3_nearfar_v2_r0.6_s2_nohis_ep5_20260827_172317/epoch_5.pth}"

EPOCHS="${EPOCHS:-5}"
BATCH_SIZE="${BATCH_SIZE:-6}"
TOTAL_BATCH_SIZE="${TOTAL_BATCH_SIZE:-24}"
IMAGE_WIDTH="${IMAGE_WIDTH:-512}"
IMAGE_HEIGHT="${IMAGE_HEIGHT:-288}"
SEED="${SEED:-0}"

# Set either switch to 0 to skip a completed second-stage run.
RUN_ADHR_BASE="${RUN_ADHR_BASE:-1}"
RUN_NEARFAR_BASE="${RUN_NEARFAR_BASE:-1}"
NEARFAR_NEAR_RATIO="${NEARFAR_NEAR_RATIO:-0.6}"
NEARFAR_FAR_STRIDE="${NEARFAR_FAR_STRIDE:-2}"

GAP_REFINER_CHANNELS="${GAP_REFINER_CHANNELS:-24}"
GAP_REFINER_BLOCKS="${GAP_REFINER_BLOCKS:-3}"
GAP_REFINER_IMAGE_ACTIVE_RATIO="${GAP_REFINER_IMAGE_ACTIVE_RATIO:-0.08}"
GAP_REFINER_IMAGE_CHANNELS="${GAP_REFINER_IMAGE_CHANNELS:-24}"
GAP_REFINER_IMAGE_LEVELS="${GAP_REFINER_IMAGE_LEVELS:-2}"
GAP_REFINER_IMAGE_CROP_RATIO="${GAP_REFINER_IMAGE_CROP_RATIO:-0.5}"

for required_path in "${BASE_CONFIG}" "${ADHR_BASE_CHECKPOINT}" \
                     "${NEARFAR_BASE_CHECKPOINT}"; do
  if [[ ! -f "${required_path}" ]]; then
    echo "Missing required file: ${required_path}" >&2
    exit 1
  fi
done

run_gapref_stage2() {
  local name="$1"
  local checkpoint="$2"
  local use_nearfar="$3"
  local work_dir="work_dirs/front3_gapref_stage2_${name}_ep${EPOCHS}_$(date +%Y%m%d_%H%M%S)"

  echo "[${name}] frozen base; training only image-guided GapRef"
  PYTHONPATH="$(pwd):${PYTHONPATH:-}" CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}" \
    "${PYTHON_BIN}" -m torch.distributed.run --standalone \
    --nproc_per_node="${NUM_GPUS}" tools/train.py "${BASE_CONFIG}" \
    --launcher pytorch --deterministic --seed "${SEED}" \
    --work-dir "${work_dir}" --load-from "${checkpoint}" \
    --num-gpus "${NUM_GPUS}" --data-root "${DATA_ROOT}" \
    --val-ann-file "${VAL_ANN_FILE}" --batch-size "${BATCH_SIZE}" \
    --total-batch-size "${TOTAL_BATCH_SIZE}" --image-width "${IMAGE_WIDTH}" \
    --image-height "${IMAGE_HEIGHT}" --epochs "${EPOCHS}" \
    --use-fp16 1 --use-tghd 0 --use-acfs-bev 0 \
    --use-row-topology 0 --use-crop-gap-refinement 0 \
    --use-selective-c2f 0 --use-dual-hardness-refinement 0 \
    --use-fixed-group-decoder 0 \
    --use-nearfar-bev "${use_nearfar}" \
    --nearfar-near-ratio "${NEARFAR_NEAR_RATIO}" \
    --nearfar-far-stride "${NEARFAR_FAR_STRIDE}" \
    --history-frames 0 --predict-future-occ 0 --future-occ-steps 0 \
    --predict-future-traj 0 --future-traj-steps 6 \
    --use-gap-residual-refiner 1 --freeze-gap-refiner-base \
    --gap-refiner-channels "${GAP_REFINER_CHANNELS}" \
    --gap-refiner-blocks "${GAP_REFINER_BLOCKS}" \
    --gap-refiner-use-bev-feature 1 \
    --gap-refiner-use-image-features 1 \
    --gap-refiner-image-active-ratio "${GAP_REFINER_IMAGE_ACTIVE_RATIO}" \
    --gap-refiner-image-channels "${GAP_REFINER_IMAGE_CHANNELS}" \
    --gap-refiner-image-levels "${GAP_REFINER_IMAGE_LEVELS}" \
    --gap-refiner-image-crop-ratio "${GAP_REFINER_IMAGE_CROP_RATIO}" \
    --gap-refiner-coarse-loss-weight 0 \
    --gap-refiner-boundary-loss-weight 0 \
    --gap-refiner-gap-loss-weight 0 \
    --gap-refiner-crop-loss-weight 0
}

# Stage 2A: start from the ADHR base but disable its training-only losses.
if [[ "${RUN_ADHR_BASE}" == "1" ]]; then
  run_gapref_stage2 "adhr_base_image_aux0" "${ADHR_BASE_CHECKPOINT}" 0
fi

# Stage 2B: preserve the NearFar encoder used by the loaded base checkpoint.
if [[ "${RUN_NEARFAR_BASE}" == "1" ]]; then
  run_gapref_stage2 "nearfar_r${NEARFAR_NEAR_RATIO}_s${NEARFAR_FAR_STRIDE}_image_aux0" \
    "${NEARFAR_BASE_CHECKPOINT}" 1
fi
