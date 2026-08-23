#!/usr/bin/env bash
set -euo pipefail

# Edit only these two values for the desired visible GPU IDs and process count.
CUDA_VISIBLE_DEVICES="0"
NUM_GPUS=1
# Per-GPU batch size. Total batch size is BATCH_SIZE * NUM_GPUS.
BATCH_SIZE=2
# Number of complete passes over the training set.
EPOCHS=2
# Number of historical images in each sample (plus the current image).
HISTORY_FRAMES=2
# Set to 1 to train/evaluate future occupancy; FUTURE_OCC_STEPS is ignored when 0.
PREDICT_FUTURE_OCC=0
FUTURE_OCC_STEPS=0
# Set to 1 to enable trajectory prediction; FarmSim trajectories are derived from poses.
PREDICT_FUTURE_TRAJ=0
FUTURE_TRAJ_STEPS=6
# Override this only if dow2 was created in a different conda root.
PYTHON_BIN="${PYTHON_BIN:-/home/hl/miniconda3/envs/dow2/bin/python}"

WORK_DIR="work_dirs/front3_$(date +%Y%m%d_%H%M%S)"
CONFIG="projects/configs/farmsim/farmsim_occ_front3.py"

if ! [[ "${BATCH_SIZE}" =~ ^[1-9][0-9]*$ && "${EPOCHS}" =~ ^[1-9][0-9]*$ && \
       "${HISTORY_FRAMES}" =~ ^[1-9][0-9]*$ && "${FUTURE_OCC_STEPS}" =~ ^[0-9]+$ && \
       "${FUTURE_TRAJ_STEPS}" =~ ^[0-9]+$ && \
       "${PREDICT_FUTURE_OCC}" =~ ^[01]$ && "${PREDICT_FUTURE_TRAJ}" =~ ^[01]$ ]]; then
  echo "BATCH_SIZE/EPOCHS/HISTORY_FRAMES must be positive; future occupancy/trajectory steps may be zero; prediction switches must be 0 or 1." >&2
  exit 2
fi
if [[ "${PREDICT_FUTURE_OCC}" -eq 1 ]]; then OCC_STEPS="${FUTURE_OCC_STEPS}"; else OCC_STEPS=0; fi
if [[ "${PREDICT_FUTURE_TRAJ}" -eq 1 ]]; then TRAJ_ENABLED=True; else TRAJ_ENABLED=False; fi
if [[ "${OCC_STEPS}" -lt 1 && "${PREDICT_FUTURE_OCC}" -eq 1 ]]; then
  echo "FUTURE_OCC_STEPS must be at least 1 when PREDICT_FUTURE_OCC=1." >&2
  exit 2
fi
if [[ "${FUTURE_TRAJ_STEPS}" -lt 1 && "${PREDICT_FUTURE_TRAJ}" -eq 1 ]]; then
  echo "FUTURE_TRAJ_STEPS must be at least 1 when PREDICT_FUTURE_TRAJ=1." >&2
  exit 2
fi
CFG_OPTIONS=("data.samples_per_gpu=${BATCH_SIZE}" "data.val.samples_per_gpu=${BATCH_SIZE}" "data.test.samples_per_gpu=${BATCH_SIZE}" \
  "total_epochs=${EPOCHS}" "runner.max_epochs=${EPOCHS}" \
  "data.train.queue_length=${HISTORY_FRAMES}" "data.val.queue_length=${HISTORY_FRAMES}" \
  "data.test.queue_length=${HISTORY_FRAMES}" \
  "data.train.future_pred_frame_num=${OCC_STEPS}" "data.val.future_pred_frame_num=${OCC_STEPS}" \
  "data.test.future_pred_frame_num=${OCC_STEPS}" \
  "data.train.future_traj_frame_num=${FUTURE_TRAJ_STEPS}" \
  "data.val.future_traj_frame_num=${FUTURE_TRAJ_STEPS}" \
  "data.test.future_traj_frame_num=${FUTURE_TRAJ_STEPS}" \
  "data.train.predict_trajectory=${TRAJ_ENABLED}" "data.val.predict_trajectory=${TRAJ_ENABLED}" \
  "data.test.predict_trajectory=${TRAJ_ENABLED}" "model.future_pred_frame_num=${OCC_STEPS}" \
  "model.test_future_frame_num=${OCC_STEPS}" "model.future_pred_head.history_queue_length=${HISTORY_FRAMES}" \
  "model.turn_on_plan=${TRAJ_ENABLED}" "model.predict_trajectory=${TRAJ_ENABLED}" \
  "model.plan_head.planning_steps=${FUTURE_TRAJ_STEPS}")

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
    "${PYTHON_BIN}" tools/train.py "${CONFIG}" --work-dir "${WORK_DIR}" --launcher none --deterministic \
    --cfg-options "${CFG_OPTIONS[@]}"
else
  PYTHONPATH="$(pwd):${PYTHONPATH:-}" \
    "${PYTHON_BIN}" -m torch.distributed.run --standalone --nproc_per_node="${NUM_GPUS}" \
    tools/train.py "${CONFIG}" --work-dir "${WORK_DIR}" --launcher pytorch --deterministic \
    --cfg-options "${CFG_OPTIONS[@]}"
fi
