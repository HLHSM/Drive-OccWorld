#!/usr/bin/env bash
# ORAD-3D monocular current-occupancy experiments:
#   1) ORAD 100% without FarmSim pretraining
#   2) FarmSim pretraining -> direct ORAD head replacement -> 10/25/50/100%
#
# Examples (all enabled runs use one shared seed and the same validation/test):
#   RUN_SCRATCH_100=1 bash tools/train_orad3d_ablation.sh
#   FARMSIM_CHECKPOINT=work_dirs/<farmsim-run>/epoch_8.pth \
#     RUN_FINETUNE_10=1 RUN_FINETUNE_25=1 RUN_FINETUNE_50=1 \
#     RUN_FINETUNE_100=1 bash tools/train_orad3d_ablation.sh
#   TEST_ANN_FILE=data/orad3d/splits/test_farm.json RUN_TEST=1 \
#     RUN_SCRATCH_100=1 bash tools/train_orad3d_ablation.sh
#
# No training is enabled by default: set the RUN_* switches above explicitly.
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
NUM_GPUS="${NUM_GPUS:-4}"
ORAD_ROOT="${ORAD_ROOT:-/data/HL/ORAD-3D/extracted}"
PYTHON_BIN="${PYTHON_BIN:-/home/HL/.conda/envs/dow2/bin/python}"
CONFIG="projects/configs/orad3d/orad3d_occ_mono.py"
SPLIT_DIR="data/orad3d/splits"
TEST_ANN_FILE="${TEST_ANN_FILE:-${SPLIT_DIR}/test.json}"
GENERIC_PRETRAINED="${GENERIC_PRETRAINED:-$(pwd)/pretrained/r101_dcn_fcos3d_pretrain.pth}"
FARMSIM_CHECKPOINT="${FARMSIM_CHECKPOINT:-work_dirs/front3_gvad_adhr_nearfar_r0.6_s2_nohis_ep8_20260904_163459/epoch_8.pth}"

BATCH_SIZE="${BATCH_SIZE:-6}"
TOTAL_BATCH_SIZE="${TOTAL_BATCH_SIZE:-24}"
WORKERS_PER_GPU="${WORKERS_PER_GPU:-4}"
IMAGE_WIDTH="${IMAGE_WIDTH:-512}"
IMAGE_HEIGHT="${IMAGE_HEIGHT:-288}"
EPOCHS="${EPOCHS:-8}"
SEED="${SEED:-20260904}"
USE_FP16="${USE_FP16:-1}"

PREPARE_SPLITS="${PREPARE_SPLITS:-1}"
AUDIT_LABELS="${AUDIT_LABELS:-0}"
RUN_SCRATCH_100="${RUN_SCRATCH_100:-1}"
RUN_FINETUNE_10="${RUN_FINETUNE_10:-1}"
RUN_FINETUNE_25="${RUN_FINETUNE_25:-1}"
RUN_FINETUNE_50="${RUN_FINETUNE_50:-1}"
RUN_FINETUNE_100="${RUN_FINETUNE_100:-1}"
RUN_TEST="${RUN_TEST:-1}"

[[ -d "${ORAD_ROOT}/training" && -d "${ORAD_ROOT}/validation" && \
   -d "${ORAD_ROOT}/testing" ]] || {
  echo "Missing extracted ORAD-3D splits under ${ORAD_ROOT}" >&2
  exit 1
}
if [[ "${RUN_SCRATCH_100}" == "1" ]]; then
  [[ -f "${GENERIC_PRETRAINED}" ]] || {
    echo "Missing generic image checkpoint: ${GENERIC_PRETRAINED}" >&2
    exit 1
  }
fi

if [[ "${PREPARE_SPLITS}" == "1" ]]; then
  prepare_args=(
    tools/prepare_orad3d.py
    --data-root "${ORAD_ROOT}"
    --output-dir "${SPLIT_DIR}"
    --seed "${SEED}"
  )
  if [[ "${AUDIT_LABELS}" != "1" ]]; then
    prepare_args+=(--skip-label-audit)
  fi
  "${PYTHON_BIN}" "${prepare_args[@]}"
fi

for manifest in train_010 train_025 train_050 train_100 val; do
  [[ -f "${SPLIT_DIR}/${manifest}.json" ]] || {
    echo "Missing ORAD manifest: ${SPLIT_DIR}/${manifest}.json" >&2
    exit 1
  }
done
[[ -f "${TEST_ANN_FILE}" ]] || {
  echo "Missing ORAD test manifest: ${TEST_ANN_FILE}" >&2
  exit 1
}

needs_farmsim=0
for enabled in "${RUN_FINETUNE_10}" "${RUN_FINETUNE_25}" \
               "${RUN_FINETUNE_50}" "${RUN_FINETUNE_100}"; do
  [[ "${enabled}" == "1" ]] && needs_farmsim=1
done

if [[ "${RUN_SCRATCH_100}" != "1" && "${needs_farmsim}" != "1" ]]; then
  echo "No experiment is enabled. Set RUN_SCRATCH_100=1 and/or RUN_FINETUNE_{10,25,50,100}=1." >&2
  exit 2
fi

adapted_checkpoint=''
if [[ "${needs_farmsim}" == "1" ]]; then
  [[ -n "${FARMSIM_CHECKPOINT}" && -f "${FARMSIM_CHECKPOINT}" ]] || {
    echo "Set FARMSIM_CHECKPOINT to a completed FarmSim checkpoint." >&2
    exit 1
  }
  adapt_tag="$(date +%Y%m%d_%H%M%S)"
  adapted_checkpoint="work_dirs/orad3d_adapted_farmsim_${adapt_tag}.pth"
  PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}" "${PYTHON_BIN}" \
    tools/adapt_farmsim_checkpoint_to_orad3d.py \
    --source "${FARMSIM_CHECKPOINT}" \
    --config "${CONFIG}" \
    --output "${adapted_checkpoint}"
fi

evaluate_checkpoint() {
  local checkpoint="$1"
  local output_dir="$2"
  PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}" CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}" \
    "${PYTHON_BIN}" -m torch.distributed.run --standalone \
    --nproc_per_node="${NUM_GPUS}" tools/test.py "${CONFIG}" "${checkpoint}" \
    --launcher pytorch --batch-size "${BATCH_SIZE}" \
    --out "${output_dir}/test_metrics.pkl" \
    --cfg-options \
      data.test.data_root="${ORAD_ROOT}" \
      data.test.ann_file="${TEST_ANN_FILE}" \
      data.test.image_size="[${IMAGE_WIDTH},${IMAGE_HEIGHT}]" \
      data.test.queue_length=0 \
      model.future_pred_head.history_queue_length=0
}

train_one() {
  local name="$1"
  local fraction="$2"
  local initialization="$3"
  local work_dir="work_dirs/orad3d_${name}_p${fraction}_ep${EPOCHS}_$(date +%Y%m%d_%H%M%S)"
  local train_manifest="${SPLIT_DIR}/train_$(printf '%03d' "${fraction}").json"

  PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}" CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}" \
    "${PYTHON_BIN}" -m torch.distributed.run --standalone \
    --nproc_per_node="${NUM_GPUS}" tools/train.py "${CONFIG}" \
    --launcher pytorch --deterministic --seed "${SEED}" \
    --work-dir "${work_dir}" --load-from "${initialization}" \
    --num-gpus "${NUM_GPUS}" --data-root "${ORAD_ROOT}" \
    --train-ann-file "${train_manifest}" \
    --val-ann-file "${SPLIT_DIR}/val.json" \
    --workers-per-gpu "${WORKERS_PER_GPU}" \
    --batch-size "${BATCH_SIZE}" --total-batch-size "${TOTAL_BATCH_SIZE}" \
    --image-width "${IMAGE_WIDTH}" --image-height "${IMAGE_HEIGHT}" \
    --use-fp16 "${USE_FP16}" \
    --history-frames 0 --predict-future-occ 0 --future-occ-steps 0 \
    --predict-future-traj 0 --future-traj-steps 6 --epochs "${EPOCHS}"

  local checkpoint="${work_dir}/epoch_${EPOCHS}.pth"
  [[ -f "${checkpoint}" ]] || {
    echo "Training finished without expected checkpoint: ${checkpoint}" >&2
    exit 1
  }
  if [[ "${RUN_TEST}" == "1" ]]; then
    evaluate_checkpoint "${checkpoint}" "${work_dir}"
  fi
}

if [[ "${RUN_SCRATCH_100}" == "1" ]]; then
  train_one scratch 100 "${GENERIC_PRETRAINED}"
fi

EPOCHS=4
[[ "${RUN_FINETUNE_10}" == "1" ]] && train_one farmsim_ft 10 "${adapted_checkpoint}"
[[ "${RUN_FINETUNE_25}" == "1" ]] && train_one farmsim_ft 25 "${adapted_checkpoint}"
[[ "${RUN_FINETUNE_50}" == "1" ]] && train_one farmsim_ft 50 "${adapted_checkpoint}"
[[ "${RUN_FINETUNE_100}" == "1" ]] && train_one farmsim_ft 100 "${adapted_checkpoint}"
