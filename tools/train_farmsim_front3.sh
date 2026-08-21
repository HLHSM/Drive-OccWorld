#!/usr/bin/env bash
set -euo pipefail

# Edit only these two values for the desired visible GPU IDs and process count.
CUDA_VISIBLE_DEVICES="0,1"
NUM_GPUS=2

CONFIG="projects/configs/farmsim/farmsim_occ_front3.py"
PORT="${PORT:-29502}"

IFS=',' read -r -a GPU_IDS <<< "${CUDA_VISIBLE_DEVICES}"
if [[ ${#GPU_IDS[@]} -ne ${NUM_GPUS} ]]; then
  echo "NUM_GPUS (${NUM_GPUS}) must equal the number of CUDA_VISIBLE_DEVICES entries (${#GPU_IDS[@]})." >&2
  exit 2
fi

export CUDA_VISIBLE_DEVICES
PYTHONPATH="$(pwd):${PYTHONPATH:-}" \
python -m torch.distributed.launch --nproc_per_node="${NUM_GPUS}" --master_port="${PORT}" \
  tools/train.py "${CONFIG}" --launcher pytorch --deterministic --no-validate
