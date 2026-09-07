#!/usr/bin/env bash
# Evaluate the nine requested ORAD experiments on farm_all and official test.
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

PYTHON_BIN="${PYTHON_BIN:-/home/HL/.conda/envs/dow2/bin/python}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
NUM_GPUS="${NUM_GPUS:-4}"
BATCH_SIZE="${BATCH_SIZE:-3}"
ORAD_ROOT="${ORAD_ROOT:-/data/HL/ORAD-3D/extracted}"
OUTPUT_DIR="${OUTPUT_DIR:-work_dirs/orad3d_evaluations}"

args=()
[[ "${FORCE:-0}" == "1" ]] && args+=(--force)
"${PYTHON_BIN}" tools/evaluate_orad3d_experiments.py \
  --repo-root "${REPO_ROOT}" --python-bin "${PYTHON_BIN}" \
  --cuda-visible-devices "${CUDA_VISIBLE_DEVICES}" --num-gpus "${NUM_GPUS}" \
  --batch-size "${BATCH_SIZE}" --orad-root "${ORAD_ROOT}" \
  --output-dir "${OUTPUT_DIR}" "${args[@]}"
