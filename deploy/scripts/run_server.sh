#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "${ROOT}"
export PATH="${ROOT}/deploy/.venv/bin:/usr/local/cuda/bin:${PATH}"
export PYTHONPATH="${ROOT}:${PYTHONPATH:-}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
PYTHON="${ROOT}/deploy/.venv/bin/python"
SAMPLES="${SAMPLES:-100}"
CALIBRATION_SAMPLES="${CALIBRATION_SAMPLES:-128}"
mkdir -p deploy/artifacts/logs
bash deploy/scripts/build_plugins.sh
"${PYTHON}" -m deploy.verify 2>&1 | tee deploy/artifacts/logs/verify.log
"${PYTHON}" -m deploy.export 2>&1 | tee deploy/artifacts/logs/export.log
"${PYTHON}" -m deploy.build_engine --precision fp16 2>&1 | tee deploy/artifacts/logs/build_fp16.log
"${PYTHON}" -m deploy.quantize --samples "${CALIBRATION_SAMPLES}" 2>&1 | tee deploy/artifacts/logs/quantize.log
"${PYTHON}" -m deploy.build_engine --precision int8 --output deploy/artifacts/int8 2>&1 | tee deploy/artifacts/logs/build_int8.log
"${PYTHON}" -m deploy.evaluate --backend original --samples "${SAMPLES}" --output deploy/artifacts/original.json
"${PYTHON}" -m deploy.evaluate --backend cuda --samples "${SAMPLES}" --compare-original --output deploy/artifacts/cuda.json
"${PYTHON}" -m deploy.evaluate --backend trt --samples "${SAMPLES}" --compare-original --output deploy/artifacts/fp16.json
"${PYTHON}" -m deploy.evaluate --backend trt --engine-dir deploy/artifacts/int8 --samples "${SAMPLES}" --compare-original --output deploy/artifacts/int8.json
