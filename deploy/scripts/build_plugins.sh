#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
ARCH="${CUDA_ARCH:-$(uname -m | sed 's/x86_64/89/;s/aarch64/87/')}"
DEPLOY_PYTHON="${DEPLOY_PYTHON:-${ROOT}/deploy/.venv/bin/python}"
if [[ "$(uname -m)" == aarch64 ]]; then
  TRT_INCLUDE="${TRT_INCLUDE:-/usr/include/aarch64-linux-gnu}"
  TRT_LIB="${TRT_LIB:-/usr/lib/aarch64-linux-gnu}"
else
  TRT_INCLUDE="${TRT_INCLUDE:-${ROOT}/deploy/build/TensorRT/include}"
  TRT_LIB="${TRT_LIB:-$(${DEPLOY_PYTHON} -c 'import importlib.util; from pathlib import Path; print(Path(importlib.util.find_spec("tensorrt_libs").origin).parent)')}"
fi
cmake -S "${ROOT}/deploy/trt_plugins" -B "${ROOT}/deploy/build/plugins" \
  -DCMAKE_CUDA_COMPILER="${CUDA_HOME:-/usr/local/cuda}/bin/nvcc" \
  -DCMAKE_CUDA_ARCHITECTURES="${ARCH}" \
  -DTENSORRT_INCLUDE_DIR="${TRT_INCLUDE}" -DTENSORRT_LIBRARY_DIR="${TRT_LIB}"
cmake --build "${ROOT}/deploy/build/plugins" --parallel 2
