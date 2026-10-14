#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
[[ "$(uname -m)" == aarch64 ]] || { echo 'Run this script on Jetson ARM64'; exit 1; }
# JetPack supplies TensorRT and CUDA. Do not replace these with server wheels.
python3 -c 'import tensorrt; print("Device TensorRT:",tensorrt.__version__)'
python3 -m venv --system-site-packages "${ROOT}/deploy/.venv"
"${ROOT}/deploy/.venv/bin/python" -m pip install 'numpy==1.23.5'
CUDA_ARCH=87 DEPLOY_PYTHON="${ROOT}/deploy/.venv/bin/python" bash "${ROOT}/deploy/scripts/build_plugins.sh"
