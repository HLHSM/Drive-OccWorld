#!/usr/bin/env bash
# Paper resolution table only; shared x=[0,15], y=[-7.5,7.5] ROI:
# 1) scratch with image pretraining: native 150x150 BEV, 8 epochs;
# 2) frozen source + interpolated 150x150 query/position tables, 8 epochs;
# 3) frozen source 75x75 BEV + trainable 2x occupancy head, 8 epochs.
# Source: R101+TSA+Agri-AMoE96, cropsample 0.1/min500. All target runs
# keep uniform sampling. DRY_RUN=1 prints commands without launching jobs.
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-2,3}"
IFS=',' read -r -a visible_devices <<< "${CUDA_VISIBLE_DEVICES}"
NUM_GPUS="${NUM_GPUS:-${#visible_devices[@]}}"
PYTHON_BIN="${PYTHON_BIN:-/home/HL/.conda/envs/dow2/bin/python}"
DATA_ROOT="${DATA_ROOT:-/data/HL/SimData-Occ0.1/SimData-Occ0.1}"
SPLIT_DIR="${SPLIT_DIR:-data/simdata_occ0p1/splits}"
CONFIG="${CONFIG:-projects/configs/farmsim/farmsim_occ_front3_s_0p1_tsa_amoe.py}"
HEADREFINE_CONFIG="${HEADREFINE_CONFIG:-projects/configs/farmsim/farmsim_occ_front3_s_0p1_headrefine_tsa_amoe.py}"
SOURCE_02_CHECKPOINT="${SOURCE_02_CHECKPOINT:-work_dirs/front3_s_base_agri_amoe96_cropsample0.1_min500_nohis_b24_ep15_20261012_214738/epoch_15.pth}"
IMAGE_PRETRAINED="${IMAGE_PRETRAINED:-pretrained/r101_dcn_fcos3d_pretrain.pth}"

BATCH_SIZE="${BATCH_SIZE:-1}"
# Identical micro-batches by default; no hidden BATCH_SIZE=6 reassignment.
HEADREFINE_BATCH_SIZE="${HEADREFINE_BATCH_SIZE:-${BATCH_SIZE}}"
TOTAL_BATCH_SIZE="${TOTAL_BATCH_SIZE:-24}"
WORKERS_PER_GPU="${WORKERS_PER_GPU:-4}"
IMAGE_WIDTH="${IMAGE_WIDTH:-512}"
IMAGE_HEIGHT="${IMAGE_HEIGHT:-288}"
EPOCHS="${EPOCHS:-15}"
SEED="${SEED:-20260912}"
USE_FP16="${USE_FP16:-1}"
RUN_SCRATCH="${RUN_SCRATCH:-1}"
RUN_FROZEN_FINETUNE="${RUN_FROZEN_FINETUNE:-1}"
RUN_FROZEN_HEADREFINE="${RUN_FROZEN_HEADREFINE:-1}"
DRY_RUN="${DRY_RUN:-0}"
RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)}"
ADAPTED_CHECKPOINT="work_dirs/checkpoint_adapters/front3_s_tsa_amoe_0p2_to_0p1_${RUN_ID}.pth"

run_command() {
  if [[ "${DRY_RUN}" == "1" ]]; then
    printf '[dry-run] '; printf '%q ' "$@"; printf '\n'
  else
    "$@"
  fi
}

if (( NUM_GPUS < 1 || NUM_GPUS > ${#visible_devices[@]} )); then
  echo 'NUM_GPUS must be positive and must not exceed the visible GPU count.' >&2; exit 2
fi
for batch in "${BATCH_SIZE}" "${HEADREFINE_BATCH_SIZE}"; do
  if (( batch < 1 || TOTAL_BATCH_SIZE < batch * NUM_GPUS ||
        TOTAL_BATCH_SIZE % (batch * NUM_GPUS) != 0 )); then
    echo 'Total batch must be a positive multiple of each per-GPU batch x GPU count.' >&2; exit 2
  fi
done
if [[ "${RUN_SCRATCH}" != "1" && "${RUN_FROZEN_FINETUNE}" != "1" &&
      "${RUN_FROZEN_HEADREFINE}" != "1" ]]; then
  echo 'No paper experiment enabled; set one of the three RUN_* switches to 1.' >&2; exit 2
fi
[[ -d "${DATA_ROOT}" ]] || { echo "Missing data root: ${DATA_ROOT}" >&2; exit 1; }
for path in "${SPLIT_DIR}/train.json" "${SPLIT_DIR}/val.json" "${CONFIG}" "${HEADREFINE_CONFIG}"; do
  [[ -f "${path}" ]] || { echo "Missing file: ${path}" >&2; exit 1; }
done
if [[ "${RUN_SCRATCH}" == "1" && ! -f "${IMAGE_PRETRAINED}" ]]; then
  echo "Missing image checkpoint: ${IMAGE_PRETRAINED}" >&2; exit 1
fi
if [[ "${RUN_FROZEN_FINETUNE}" == "1" || "${RUN_FROZEN_HEADREFINE}" == "1" ]]; then
  [[ -f "${SOURCE_02_CHECKPOINT}" ]] || {
    echo "Missing 0.2 m source checkpoint: ${SOURCE_02_CHECKPOINT}" >&2; exit 1;
  }
fi
if [[ "${RUN_FROZEN_FINETUNE}" == "1" ]]; then
  # Always construct a fresh adapter for this source/config/seed combination.
  run_command env PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}" "${PYTHON_BIN}" \
    tools/adapt_farmsim_checkpoint_to_0p1.py \
    --source "${SOURCE_02_CHECKPOINT}" --config "${CONFIG}" \
    --output "${ADAPTED_CHECKPOINT}" --seed "${SEED}"
fi

train_one() {
  local name="$1" config="$2" initialization="$3" batch="$4"
  shift 4
  local work_dir="work_dirs/simdata_occ0p1_s_tsa_amoe_${name}_b${TOTAL_BATCH_SIZE}_ep${EPOCHS}_${RUN_ID}"
  echo "Starting ${name}: ROI 15x15 m; 150x150x25 output; batch/GPU=${batch}, effective batch=${TOTAL_BATCH_SIZE}"
  run_command env PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}" CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}" \
    "${PYTHON_BIN}" -m torch.distributed.run --standalone \
    --nproc_per_node="${NUM_GPUS}" tools/train.py "${config}" \
    --launcher pytorch --deterministic --seed "${SEED}" --work-dir "${work_dir}" \
    --load-from "${initialization}" \
    --num-gpus "${NUM_GPUS}" --data-root "${DATA_ROOT}" \
    --train-ann-file "${SPLIT_DIR}/train.json" --val-ann-file "${SPLIT_DIR}/val.json" \
    --workers-per-gpu "${WORKERS_PER_GPU}" \
    --batch-size "${batch}" --total-batch-size "${TOTAL_BATCH_SIZE}" \
    --image-width "${IMAGE_WIDTH}" --image-height "${IMAGE_HEIGHT}" \
    --use-fp16 "${USE_FP16}" --history-frames 0 \
    --predict-future-occ 0 --future-occ-steps 0 \
    --predict-future-traj 0 --future-traj-steps 6 --epochs "${EPOCHS}" \
    --disable-temporal-self-attention 0 --use-gvad-attention 0 \
    --use-gvadv2-attention 0 --use-nearfar-bev 0 \
    --use-dual-hardness-refinement 0 --use-agri-amoe-decoder 1 \
    --agri-amoe-channels 96 --use-crop-positive-sampling 0 \
    "$@" --cfg-options evaluation.interval=1 checkpoint_config.interval=1
}

if [[ "${RUN_SCRATCH}" == "1" ]]; then
  train_one scratch_image_pretrained "${CONFIG}" "${IMAGE_PRETRAINED}" "${BATCH_SIZE}"
fi
if [[ "${RUN_FROZEN_FINETUNE}" == "1" ]]; then
  train_one query_adapt_frozen_base "${CONFIG}" "${ADAPTED_CHECKPOINT}" "${BATCH_SIZE}" \
    --freeze-resolution-transfer-base
fi
if [[ "${RUN_FROZEN_HEADREFINE}" == "1" ]]; then
  train_one headrefine_frozen_base "${HEADREFINE_CONFIG}" "${SOURCE_02_CHECKPOINT}" \
    "${HEADREFINE_BATCH_SIZE}" --freeze-resolution-transfer-base
fi
