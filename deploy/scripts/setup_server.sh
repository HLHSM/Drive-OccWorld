#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
BASE_PYTHON="${BASE_PYTHON:-/home/HL/.conda/envs/dow2/bin/python}"
"${BASE_PYTHON}" -m venv --system-site-packages "${ROOT}/deploy/.venv"
"${ROOT}/deploy/.venv/bin/python" -m pip install -r "${ROOT}/deploy/requirements-server.txt"
# Only headers are used. Orin uses its own JetPack-installed headers/libraries.
if [[ ! -f "${ROOT}/deploy/build/TensorRT/include/NvInfer.h" ]]; then
  git clone --depth 1 --branch release/10.3 https://github.com/NVIDIA/TensorRT.git "${ROOT}/deploy/build/TensorRT"
fi
