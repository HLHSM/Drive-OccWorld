#!/usr/bin/env bash
# Two tiny epochs through the real SparseOcc runner, including both evals.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REPO="${ROOT}/experiment/official/SparseOcc"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
IFS=',' read -r -a devices <<< "${CUDA_VISIBLE_DEVICES}"
NUM_GPUS="${NUM_GPUS:-${#devices[@]}}"
CUDA118_HOME="${CUDA118_HOME:-/home/HL/.local/cuda-11.8}"
SPARSE_ENV="${SPARSE_ENV:-sparseocc}"
RUN_NAME="${SPARSE_RUN_NAME:-farmsim_front3_s_geomfix_smoke_$(date +%Y%m%d_%H%M%S)_$$}"
if [[ -e "${REPO}/outputs/SparseOcc/${RUN_NAME}" ]]; then
  echo "Refusing to overwrite existing output: ${RUN_NAME}" >&2; exit 2
fi
export CUDA_VISIBLE_DEVICES
export PYTHONPATH="${REPO}"
export DRIVE_OCCWORLD_ROOT="${ROOT}"
export FARMSIM_DATA_ROOT="${DATA_ROOT:-/data/HL/SimData-Occ/SimData}"
export FARMSIM_INFOS_ROOT="${FARMSIM_INFOS_ROOT:-${ROOT}/experiment/common/infos_front3_s}"
export FARMSIM_BATCH_SIZE=1 FARMSIM_WORLD_SIZE="${NUM_GPUS}"
export FARMSIM_GRAD_ACCUM_STEPS=2 FARMSIM_USE_FP16=1
export FARMSIM_IMAGE_WIDTH=512 FARMSIM_IMAGE_HEIGHT=288 WORKERS_PER_GPU=1
export CUDA_HOME="${CUDA118_HOME}" CUDA_PATH="${CUDA118_HOME}"
export PATH="${CUDA118_HOME}/bin:${PATH}"
export LD_LIBRARY_PATH="${CUDA118_HOME}/lib64:${LD_LIBRARY_PATH:-}"
cd "${REPO}"
echo "SparseOcc smoke: GPUs=${CUDA_VISIBLE_DEVICES}; output=${REPO}/outputs/SparseOcc/${RUN_NAME}"
conda run --no-capture-output -n "${SPARSE_ENV}" python -m torch.distributed.run \
  --nnodes=1 --node_rank=0 --master_addr=127.0.0.1 \
  --master_port="${SPARSE_MASTER_PORT:-29547}" --nproc_per_node="${NUM_GPUS}" train.py \
  --config configs/farmsim_front3_current.py --run_name "${RUN_NAME}" \
  --override total_epochs=2 data.train.smoke_num_samples=8 \
  data.val.smoke_num_samples=4 data.test.smoke_num_samples=4 \
  log_config.hooks.1.interval=1
