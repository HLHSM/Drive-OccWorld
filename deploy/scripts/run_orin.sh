#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "${ROOT}"
export PYTHONPATH="${ROOT}:${PYTHONPATH:-}"
export LD_LIBRARY_PATH="/usr/local/cuda/lib64:/usr/lib/aarch64-linux-gnu:${LD_LIBRARY_PATH:-}"
PYTHON="${DEPLOY_PYTHON:-${ROOT}/deploy/.venv/bin/python}"
PRECISION="${PRECISION:-fp16}"
"${PYTHON}" -m deploy.build_engine --precision "${PRECISION}" --output "deploy/artifacts/${PRECISION}"
"${PYTHON}" -m deploy.infer --engine-dir "deploy/artifacts/${PRECISION}" \
  --plugin deploy/build/plugins/libagriocc_plugins.so --input deploy/artifacts/model/example.npz \
  --output "deploy/artifacts/${PRECISION}/example_prediction.npz"
