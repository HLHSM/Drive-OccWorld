#!/usr/bin/env bash
# Paper table only: ORAD 100% scratch (6 epochs), 100% transfer (2 epochs).
# Source: 15x15 m R101+TSA+Agri-AMoE96, cropsample 0.1/min500.
# ORAD retains its official grid/classes and uniform target-domain sampling.
# DRY_RUN=1 prints commands without adapting checkpoints or launching jobs.
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-2,3}"
IFS=',' read -r -a visible_devices <<< "${CUDA_VISIBLE_DEVICES}"
NUM_GPUS="${NUM_GPUS:-${#visible_devices[@]}}"
ORAD_ROOT="${ORAD_ROOT:-/data/HL/ORAD-3D/extracted}"
PYTHON_BIN="${PYTHON_BIN:-/home/HL/.conda/envs/dow2/bin/python}"
CONFIG="${CONFIG:-projects/configs/orad3d/orad3d_occ_mono_tsa_amoe.py}"
SPLIT_DIR="${SPLIT_DIR:-data/orad3d/splits}"
TRAIN_ANN_FILE="${TRAIN_ANN_FILE:-${SPLIT_DIR}/train_100.json}"
VAL_ANN_FILE="${VAL_ANN_FILE:-${SPLIT_DIR}/val.json}"
TEST_ANN_FILE="${TEST_ANN_FILE:-${SPLIT_DIR}/test.json}"
FARM_TEST_ANN_FILE="${FARM_TEST_ANN_FILE:-${SPLIT_DIR}/test_farm.json}"
GENERIC_PRETRAINED="${GENERIC_PRETRAINED:-${REPO_ROOT}/pretrained/r101_dcn_fcos3d_pretrain.pth}"
FARMSIM_CHECKPOINT="${FARMSIM_CHECKPOINT:-work_dirs/front3_s_base_agri_amoe96_cropsample0.1_min500_nohis_b24_ep15_20261012_214738/epoch_15.pth}"

BATCH_SIZE="${BATCH_SIZE:-4}"
TOTAL_BATCH_SIZE="${TOTAL_BATCH_SIZE:-24}"
WORKERS_PER_GPU="${WORKERS_PER_GPU:-4}"
IMAGE_WIDTH="${IMAGE_WIDTH:-512}"
IMAGE_HEIGHT="${IMAGE_HEIGHT:-288}"
SCRATCH_EPOCHS="${SCRATCH_EPOCHS:-6}"
FINETUNE_EPOCHS="${FINETUNE_EPOCHS:-2}"
SEED="${SEED:-20260904}"
USE_FP16="${USE_FP16:-1}"
RUN_SCRATCH_100="${RUN_SCRATCH_100:-1}"
RUN_FINETUNE_100="${RUN_FINETUNE_100:-1}"
RUN_TEST="${RUN_TEST:-1}"
PREPARE_SPLITS="${PREPARE_SPLITS:-0}"
AUDIT_LABELS="${AUDIT_LABELS:-0}"
DRY_RUN="${DRY_RUN:-0}"
RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)}"

run_command() {
  if [[ "${DRY_RUN}" == "1" ]]; then
    printf '[dry-run] '; printf '%q ' "$@"; printf '\n'
  else
    "$@"
  fi
}

if (( NUM_GPUS < 1 || NUM_GPUS > ${#visible_devices[@]} || BATCH_SIZE < 1 ||
      TOTAL_BATCH_SIZE < BATCH_SIZE * NUM_GPUS ||
      TOTAL_BATCH_SIZE % (BATCH_SIZE * NUM_GPUS) != 0 )); then
  echo 'Invalid GPU/batch settings: total batch must be a positive multiple of per-GPU batch x GPU count.' >&2
  exit 2
fi
if [[ "${RUN_SCRATCH_100}" != "1" && "${RUN_FINETUNE_100}" != "1" ]]; then
  echo 'No run enabled; set RUN_SCRATCH_100=1 and/or RUN_FINETUNE_100=1.' >&2
  exit 2
fi
[[ -f "${CONFIG}" ]] || { echo "Missing config: ${CONFIG}" >&2; exit 1; }
if [[ "${RUN_SCRATCH_100}" == "1" && ! -f "${GENERIC_PRETRAINED}" ]]; then
  echo "Missing image checkpoint: ${GENERIC_PRETRAINED}" >&2; exit 1
fi
if [[ "${RUN_FINETUNE_100}" == "1" && ! -f "${FARMSIM_CHECKPOINT}" ]]; then
  echo "Missing FarmSim checkpoint: ${FARMSIM_CHECKPOINT}" >&2; exit 1
fi
[[ -d "${ORAD_ROOT}/training" && -d "${ORAD_ROOT}/validation" &&
   -d "${ORAD_ROOT}/testing" ]] || { echo "Missing ORAD data: ${ORAD_ROOT}" >&2; exit 1; }

if [[ "${PREPARE_SPLITS}" == "1" ]]; then
  prepare_args=(tools/prepare_orad3d.py --data-root "${ORAD_ROOT}"
                --output-dir "${SPLIT_DIR}" --seed "${SEED}" --fractions 1.0)
  [[ "${AUDIT_LABELS}" == "1" ]] || prepare_args+=(--skip-label-audit)
  run_command "${PYTHON_BIN}" "${prepare_args[@]}"
fi
for manifest in "${TRAIN_ANN_FILE}" "${VAL_ANN_FILE}"; do
  if [[ ! -f "${manifest}" && "${PREPARE_SPLITS}" != "1" ]]; then
    echo "Missing manifest: ${manifest}; use PREPARE_SPLITS=1 to prepare official splits." >&2; exit 1
  fi
done
if [[ "${RUN_TEST}" == "1" ]]; then
  for manifest in "${TEST_ANN_FILE}" "${FARM_TEST_ANN_FILE}"; do
    [[ -f "${manifest}" ]] || { echo "Missing held-out test manifest: ${manifest}" >&2; exit 1; }
  done
fi

adapted_checkpoint="work_dirs/checkpoint_adapters/orad3d_from_front3_s_tsa_amoe_${RUN_ID}.pth"
if [[ "${RUN_FINETUNE_100}" == "1" ]]; then
  # Never reuse an adapter generated for the old GVAD/NearFar source.
  run_command env PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}" "${PYTHON_BIN}" \
    tools/adapt_farmsim_checkpoint_to_orad3d.py \
    --source "${FARMSIM_CHECKPOINT}" --config "${CONFIG}" \
    --output "${adapted_checkpoint}" --center-camera-index 1 --seed "${SEED}"
fi

evaluate_checkpoint() {
  local config="$1" checkpoint="$2" output_dir="$3" training_seconds="$4"
  local subset manifest
  for subset in test test_farm; do
    if [[ "${subset}" == "test" ]]; then manifest="${TEST_ANN_FILE}";
    else manifest="${FARM_TEST_ANN_FILE}"; fi
    # tools/test.py needs the saved training config, not the old base config.
    run_command env PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}" CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}" \
      "${PYTHON_BIN}" -m torch.distributed.run --standalone \
      --nproc_per_node="${NUM_GPUS}" tools/test.py "${config}" "${checkpoint}" \
      --launcher pytorch --batch-size "${BATCH_SIZE}" \
      --out "${output_dir}/${subset}_metrics.pkl" --cfg-options \
        data.test.data_root="${ORAD_ROOT}" data.test.ann_file="${manifest}" \
        data.test.image_size="[${IMAGE_WIDTH},${IMAGE_HEIGHT}]" \
        data.workers_per_gpu="${WORKERS_PER_GPU}"
  done
  run_command "${PYTHON_BIN}" tools/summarize_orad3d_paper_transfer.py \
    --work-dir "${output_dir}" --training-seconds "${training_seconds}"
}

train_one() {
  local name="$1" epochs="$2" initialization="$3"
  local work_dir="work_dirs/orad3d_front3_s_tsa_amoe_${name}_p100_ep${epochs}_${RUN_ID}"
  echo "Starting ${name}: TSA+Agri-AMoE96, all ORAD data, epochs=${epochs}, effective batch=${TOTAL_BATCH_SIZE}"
  run_command mkdir -p "${work_dir}"
  local start_seconds="${SECONDS}"
  run_command env PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}" CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}" \
    "${PYTHON_BIN}" -m torch.distributed.run --standalone \
    --nproc_per_node="${NUM_GPUS}" tools/train.py "${CONFIG}" \
    --launcher pytorch --deterministic --seed "${SEED}" \
    --work-dir "${work_dir}" --load-from "${initialization}" \
    --num-gpus "${NUM_GPUS}" --data-root "${ORAD_ROOT}" \
    --train-ann-file "${TRAIN_ANN_FILE}" --val-ann-file "${VAL_ANN_FILE}" \
    --workers-per-gpu "${WORKERS_PER_GPU}" \
    --batch-size "${BATCH_SIZE}" --total-batch-size "${TOTAL_BATCH_SIZE}" \
    --image-width "${IMAGE_WIDTH}" --image-height "${IMAGE_HEIGHT}" \
    --use-fp16 "${USE_FP16}" --history-frames 0 \
    --predict-future-occ 0 --future-occ-steps 0 \
    --predict-future-traj 0 --future-traj-steps 6 --epochs "${epochs}" \
    --disable-temporal-self-attention 0 --use-gvad-attention 0 \
    --use-gvadv2-attention 0 --use-nearfar-bev 0 \
    --use-dual-hardness-refinement 0 --use-agri-amoe-decoder 1 \
    --agri-amoe-channels 96 --use-crop-positive-sampling 0 \
    --cfg-options evaluation.interval=1 checkpoint_config.interval=1
  local training_seconds="$((SECONDS - start_seconds))"
  echo "${name}: training + epoch validation took ${training_seconds} seconds (excludes conversion and held-out tests)."
  local checkpoint="${work_dir}/epoch_${epochs}.pth"
  if [[ "${DRY_RUN}" != "1" && ! -f "${checkpoint}" ]]; then
    echo "Missing final checkpoint: ${checkpoint}" >&2; exit 1
  fi
  if [[ "${RUN_TEST}" == "1" ]]; then
    evaluate_checkpoint "${work_dir}/$(basename "${CONFIG}")" "${checkpoint}" "${work_dir}" "${training_seconds}"
  fi
}

if [[ "${RUN_SCRATCH_100}" == "1" ]]; then
  train_one scratch_image_pretrained "${SCRATCH_EPOCHS}" "${GENERIC_PRETRAINED}"
fi
if [[ "${RUN_FINETUNE_100}" == "1" ]]; then
  train_one farmsim_transfer "${FINETUNE_EPOCHS}" "${adapted_checkpoint}"
fi
