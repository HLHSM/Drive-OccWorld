#!/usr/bin/env bash
# End-to-end ORAD training/evaluation smoke test on GPUs 0 and 1.
# The test loader intentionally uses batch size 7: 16 samples on two ranks
# gives each rank a final one-sample batch, covering the former 3D-input bug.
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"
NUM_GPUS="${NUM_GPUS:-2}"
ORAD_ROOT="${ORAD_ROOT:-/data/HL/ORAD-3D/extracted}"
PYTHON_BIN="${PYTHON_BIN:-/home/HL/.conda/envs/dow2/bin/python}"
CONFIG="projects/configs/orad3d/orad3d_occ_mono.py"
SPLIT_DIR="data/orad3d/splits"
SMOKE_DIR="${SPLIT_DIR}/smoke48"
TRAIN_ANN_FILE="${SMOKE_DIR}/train_48.json"
TEST_ANN_FILE="${SMOKE_DIR}/test_16.json"
PRETRAINED_FROM="${PRETRAINED_FROM:-${REPO_ROOT}/pretrained/r101_dcn_fcos3d_pretrain.pth}"
WORK_DIR="${WORK_DIR:-work_dirs/orad3d_smoke48_$(date +%Y%m%d_%H%M%S)}"

BATCH_SIZE="${BATCH_SIZE:-3}"
TEST_BATCH_SIZE="${TEST_BATCH_SIZE:-7}"
TOTAL_BATCH_SIZE="${TOTAL_BATCH_SIZE:-6}"
WORKERS_PER_GPU="${WORKERS_PER_GPU:-2}"
IMAGE_WIDTH="${IMAGE_WIDTH:-512}"
IMAGE_HEIGHT="${IMAGE_HEIGHT:-288}"
USE_FP16="${USE_FP16:-1}"
EPOCHS="${EPOCHS:-1}"

[[ "${CUDA_VISIBLE_DEVICES}" == "0,1" && "${NUM_GPUS}" == "2" ]] || {
  echo 'This smoke script is intentionally fixed to CUDA devices 0,1 and two ranks.' >&2
  exit 2
}
[[ -d "${ORAD_ROOT}/training" && -d "${ORAD_ROOT}/testing" ]] || {
  echo "Missing extracted ORAD directories under ${ORAD_ROOT}" >&2
  exit 1
}
[[ -f "${PRETRAINED_FROM}" ]] || {
  echo "Missing generic image checkpoint: ${PRETRAINED_FROM}" >&2
  exit 1
}

"${PYTHON_BIN}" tools/create_orad3d_smoke_split.py \
  --split-dir "${SPLIT_DIR}" --output-dir "${SMOKE_DIR}" \
  --train-samples 48 --test-samples 16

PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}" CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}" \
  "${PYTHON_BIN}" -m torch.distributed.run --standalone \
  --nproc_per_node="${NUM_GPUS}" tools/train.py "${CONFIG}" \
  --launcher pytorch --deterministic --work-dir "${WORK_DIR}" \
  --load-from "${PRETRAINED_FROM}" --num-gpus "${NUM_GPUS}" \
  --data-root "${ORAD_ROOT}" --train-ann-file "${TRAIN_ANN_FILE}" \
  --val-ann-file "${TEST_ANN_FILE}" --batch-size "${BATCH_SIZE}" \
  --total-batch-size "${TOTAL_BATCH_SIZE}" --workers-per-gpu "${WORKERS_PER_GPU}" \
  --image-width "${IMAGE_WIDTH}" --image-height "${IMAGE_HEIGHT}" \
  --use-fp16 "${USE_FP16}" --history-frames 0 \
  --predict-future-occ 0 --future-occ-steps 0 \
  --predict-future-traj 0 --future-traj-steps 6 --epochs "${EPOCHS}"

CHECKPOINT="${WORK_DIR}/epoch_${EPOCHS}.pth"
[[ -f "${CHECKPOINT}" ]] || {
  echo "Smoke training finished without ${CHECKPOINT}" >&2
  exit 1
}

PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}" CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}" \
  "${PYTHON_BIN}" -m torch.distributed.run --standalone \
  --nproc_per_node="${NUM_GPUS}" tools/test.py "${CONFIG}" "${CHECKPOINT}" \
  --launcher pytorch --batch-size "${TEST_BATCH_SIZE}" \
  --eval mIoU \
  --out "${WORK_DIR}/test_16_metrics.pkl" --cfg-options \
    data.test.data_root="${ORAD_ROOT}" \
    data.test.ann_file="${TEST_ANN_FILE}" \
    data.test.image_size="[${IMAGE_WIDTH},${IMAGE_HEIGHT}]" \
    data.test.queue_length=0 \
    model.future_pred_head.history_queue_length=0

echo "ORAD smoke train+test passed: ${WORK_DIR}"
