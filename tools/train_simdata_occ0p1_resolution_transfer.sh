#!/usr/bin/env bash
# 0.1 m FarmSim resolution-transfer experiment.
#
# Five controlled runs use the same 0.1 m split:
#   1. 0.2 m occupancy pretraining -> BEV embedding interpolation -> fine-tune
#   2. public 2D image-backbone initialization only (no 0.2 m occupancy prior)
#   3. 0.2 m BEV encoder + a learned 2x occupancy output refiner
#   4. frozen-base variant of (1): train only the 0.1 m BEV query lattice
#   5. frozen-base variant of (3): train only the newly added 2x output head
#
# Frozen variants are opt-in, for example:
#   RUN_FINETUNE=0 RUN_SCRATCH=0 RUN_HEADREFINE=0 \
#   RUN_FROZEN_FINETUNE=1 RUN_FROZEN_HEADREFINE=1 \
#   bash tools/train_simdata_occ0p1_resolution_transfer.sh
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
NUM_GPUS="${NUM_GPUS:-4}"
PYTHON_BIN="${PYTHON_BIN:-/home/HL/.conda/envs/dow2/bin/python}"
DATA_ROOT="${DATA_ROOT:-/data/HL/SimData-Occ0.1/SimData-Occ0.1}"
SPLIT_DIR="${SPLIT_DIR:-data/simdata_occ0p1/splits}"
CONFIG="${CONFIG:-projects/configs/farmsim/farmsim_occ_front3_0p1_gvad_amoe_nearfar.py}"
HEADREFINE_CONFIG="${HEADREFINE_CONFIG:-projects/configs/farmsim/farmsim_occ_front3_0p1_headrefine_gvad_amoe_nearfar.py}"
SOURCE_02_CHECKPOINT="${SOURCE_02_CHECKPOINT:-work_dirs/front3_gvad_agri_amoe_nearfar_r0.6_s2_ep8_20260911_105909/epoch_8.pth}"
IMAGE_PRETRAINED="${IMAGE_PRETRAINED:-pretrained/r101_dcn_fcos3d_pretrain.pth}"
ADAPTED_CHECKPOINT="${ADAPTED_CHECKPOINT:-work_dirs/front3_gvad_agri_amoe_nearfar_0p1_adapter.pth}"

# 0.1 m has four times as many BEV cells as 0.2 m.  The conservative default
# keeps the micro-batch at one sample/GPU; increase only after checking memory.
BATCH_SIZE="${BATCH_SIZE:-1}"
TOTAL_BATCH_SIZE="${TOTAL_BATCH_SIZE:-24}"
WORKERS_PER_GPU="${WORKERS_PER_GPU:-4}"
IMAGE_WIDTH="${IMAGE_WIDTH:-512}"
IMAGE_HEIGHT="${IMAGE_HEIGHT:-288}"
EPOCHS="${EPOCHS:-8}"
SEED="${SEED:-20260912}"
USE_FP16="${USE_FP16:-1}"

RUN_FINETUNE="${RUN_FINETUNE:-1}"
RUN_SCRATCH="${RUN_SCRATCH:-1}"
RUN_HEADREFINE="${RUN_HEADREFINE:-1}"
RUN_FROZEN_FINETUNE="${RUN_FROZEN_FINETUNE:-1}"
RUN_FROZEN_HEADREFINE="${RUN_FROZEN_HEADREFINE:-1}"

[[ -d "${DATA_ROOT}" ]] || { echo "Missing data root: ${DATA_ROOT}" >&2; exit 1; }
for manifest in train val; do
  [[ -f "${SPLIT_DIR}/${manifest}.json" ]] || {
    echo "Missing split manifest: ${SPLIT_DIR}/${manifest}.json" >&2
    echo "Create it with tools/create_farmsim_split.py before training." >&2
    exit 1
  }
done
[[ -f "${CONFIG}" ]] || { echo "Missing config: ${CONFIG}" >&2; exit 1; }
[[ -f "${HEADREFINE_CONFIG}" ]] || { echo "Missing head-refine config: ${HEADREFINE_CONFIG}" >&2; exit 1; }

if [[ "${RUN_FINETUNE}" != "1" && "${RUN_SCRATCH}" != "1" && \
      "${RUN_HEADREFINE}" != "1" && "${RUN_FROZEN_FINETUNE}" != "1" && \
      "${RUN_FROZEN_HEADREFINE}" != "1" ]]; then
  echo "No run enabled. Set one or more RUN_* variables to 1." >&2
  exit 2
fi

if [[ ( "${RUN_FINETUNE}" == "1" || "${RUN_FROZEN_FINETUNE}" == "1" ) && \
      ! -f "${ADAPTED_CHECKPOINT}" ]]; then
  [[ -f "${SOURCE_02_CHECKPOINT}" ]] || {
    echo "Missing 0.2 m source checkpoint: ${SOURCE_02_CHECKPOINT}" >&2
    exit 1
  }
  PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}" "${PYTHON_BIN}" \
    tools/adapt_farmsim_checkpoint_to_0p1.py \
    --source "${SOURCE_02_CHECKPOINT}" --config "${CONFIG}" \
    --output "${ADAPTED_CHECKPOINT}" --seed "${SEED}"
fi

if [[ "${RUN_SCRATCH}" == "1" ]]; then
  [[ -f "${IMAGE_PRETRAINED}" ]] || {
    echo "Missing public image-backbone checkpoint: ${IMAGE_PRETRAINED}" >&2
    exit 1
  }
fi

if [[ "${RUN_HEADREFINE}" == "1" || "${RUN_FROZEN_HEADREFINE}" == "1" ]]; then
  [[ -f "${SOURCE_02_CHECKPOINT}" ]] || {
    echo "Missing 0.2 m source checkpoint: ${SOURCE_02_CHECKPOINT}" >&2
    exit 1
  }
fi

train_one() {
  local name="$1"
  local config="$2"
  local initialization="$3"
  shift 3
  local work_dir="work_dirs/simdata_occ0p1_${name}_gvad_amoe_nearfar_ep${EPOCHS}_$(date +%Y%m%d_%H%M%S)"

  PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}" CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}" \
    "${PYTHON_BIN}" -m torch.distributed.run --standalone \
    --nproc_per_node="${NUM_GPUS}" tools/train.py "${config}" \
    --launcher pytorch --deterministic --seed "${SEED}" --work-dir "${work_dir}" \
    --load-from "${initialization}" \
    --num-gpus "${NUM_GPUS}" --data-root "${DATA_ROOT}" \
    --train-ann-file "${SPLIT_DIR}/train.json" --val-ann-file "${SPLIT_DIR}/val.json" \
    --workers-per-gpu "${WORKERS_PER_GPU}" \
    --batch-size "${BATCH_SIZE}" --total-batch-size "${TOTAL_BATCH_SIZE}" \
    --image-width "${IMAGE_WIDTH}" --image-height "${IMAGE_HEIGHT}" \
    --use-fp16 "${USE_FP16}" --history-frames 0 \
    --predict-future-occ 0 --future-occ-steps 0 \
    --predict-future-traj 0 --future-traj-steps 6 --epochs "${EPOCHS}" \
    "$@"
}

# if [[ "${RUN_FINETUNE}" == "1" ]]; then
#   train_one "finetune_from_0p2" "${CONFIG}" "${ADAPTED_CHECKPOINT}"
# fi


BATCH_SIZE=1

if [[ "${RUN_SCRATCH}" == "1" ]]; then
  # This control has the same public 2D image initialization but receives no
  # 0.2 m semantic-occupancy or BEV-query pretraining.
  train_one "scratch_image_pretrained" "${CONFIG}" "${IMAGE_PRETRAINED}"
fi

BATCH_SIZE=1
if [[ "${RUN_FROZEN_FINETUNE}" == "1" ]]; then
  # Freeze all 0.2 m pretrained modules.  The adapter-created 200x200 BEV
  # query table plus row/column positional embeddings are the only updates.
  train_one "finetune_from_0p2_frozen_base" "${CONFIG}" "${ADAPTED_CHECKPOINT}" \
    --freeze-resolution-transfer-base
fi

BATCH_SIZE=6
if [[ "${RUN_FROZEN_HEADREFINE}" == "1" ]]; then
  # Freeze the unchanged 100x100 source model and learn only the new 2x
  # occupancy refinement projection/residual head.
  train_one "headrefine_from_0p2_frozen_base" "${HEADREFINE_CONFIG}" \
    "${SOURCE_02_CHECKPOINT}" --freeze-resolution-transfer-base
fi



if [[ "${RUN_HEADREFINE}" == "1" ]]; then
  # The 100x100 GVAD BEV encoder is loaded unchanged; the new learned 2x
  # occupancy refiner is initialized for the 0.1 m adaptation stage.
  train_one "headrefine_from_0p2" "${HEADREFINE_CONFIG}" "${SOURCE_02_CHECKPOINT}"
fi


