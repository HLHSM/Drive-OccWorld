#!/usr/bin/env bash
set -euo pipefail

# Edit only these two values for the desired visible GPU IDs and process count.
CUDA_VISIBLE_DEVICES="0"
NUM_GPUS=1
# Per-GPU batch size. Total batch size is BATCH_SIZE * NUM_GPUS.
BATCH_SIZE=1
# Number of complete passes over the training set.
EPOCHS=24
# Override this only if dow2 was created in a different conda root.
PYTHON_BIN="${PYTHON_BIN:-/home/hl/miniconda3/envs/dow2/bin/python}"

CONFIG="projects/configs/farmsim/farmsim_occ_front3.py"

if ! [[ "${BATCH_SIZE}" =~ ^[1-9][0-9]*$ && "${EPOCHS}" =~ ^[1-9][0-9]*$ ]]; then
  echo "BATCH_SIZE and EPOCHS must both be positive integers." >&2
  exit 2
fi
CFG_OPTIONS=("data.samples_per_gpu=${BATCH_SIZE}" "total_epochs=${EPOCHS}")

IFS=',' read -r -a GPU_IDS <<< "${CUDA_VISIBLE_DEVICES}"
if [[ ${#GPU_IDS[@]} -ne ${NUM_GPUS} ]]; then
  echo "NUM_GPUS (${NUM_GPUS}) must equal the number of CUDA_VISIBLE_DEVICES entries (${#GPU_IDS[@]})." >&2
  exit 2
fi

export CUDA_VISIBLE_DEVICES
if [[ ! -x "${PYTHON_BIN}" ]]; then
  echo "dow2 Python was not found: ${PYTHON_BIN}" >&2
  exit 2
fi
if [[ "${NUM_GPUS}" -eq 1 ]]; then
  PYTHONPATH="$(pwd):${PYTHONPATH:-}" \
    "${PYTHON_BIN}" tools/train.py "${CONFIG}" --launcher none --deterministic \
    --cfg-options "${CFG_OPTIONS[@]}"
else
  PYTHONPATH="$(pwd):${PYTHONPATH:-}" \
    "${PYTHON_BIN}" -m torch.distributed.run --standalone --nproc_per_node="${NUM_GPUS}" \
    tools/train.py "${CONFIG}" --launcher pytorch --deterministic \
    --cfg-options "${CFG_OPTIONS[@]}"
fi
