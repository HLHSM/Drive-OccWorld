#!/usr/bin/env bash
# Single-GPU inference comparison: dense basev2 on GPU 0, NearFar on GPU 1.
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-/home/HL/.conda/envs/dow2/bin/python}"
CONFIG="${CONFIG:-projects/configs/farmsim/farmsim_occ_front3.py}"
DATA_ROOT="${DATA_ROOT:-/data/HL/SimData-Occ/SimData}"
VAL_ANN_FILE="${VAL_ANN_FILE:-data/farmsim/splits/val.json}"
IMAGE_WIDTH="${IMAGE_WIDTH:-512}"
IMAGE_HEIGHT="${IMAGE_HEIGHT:-288}"
BASE_CHECKPOINT="${BASE_CHECKPOINT:-work_dirs/front3_basev2_nohis_ep5_20260827_121032/epoch_5.pth}"
NEARFAR_CHECKPOINT="${NEARFAR_CHECKPOINT:-work_dirs/front3_nearfar_v2_r0.6_s2_nohis_ep5_20260827_172317/epoch_5.pth}"
OUTPUT_DIR="${OUTPUT_DIR:-work_dirs/inference_basev2_nearfar_$(date +%Y%m%d_%H%M%S)}"

for required_path in "${CONFIG}" "${BASE_CHECKPOINT}" "${NEARFAR_CHECKPOINT}"; do
  [[ -f "${required_path}" ]] || { echo "Missing: ${required_path}" >&2; exit 1; }
done
mkdir -p "${OUTPUT_DIR}"

common_args=(
  --batch-size 1 --profile-inference --eval bbox
  --cfg-options
  model.turn_on_plan=False model.future_pred_head.use_plan_traj=False
  model.future_pred_head.history_queue_length=0
  model.future_pred_frame_num=0 model.test_future_frame_num=0
  model.future_pred_head.use_tghd=False
  model.future_pred_head.use_dual_hardness_refinement=False
  model.future_pred_head.use_gap_residual_refiner=False
  predict_trajectory=False data.test.data_root="${DATA_ROOT}"
  data.test.ann_file="${VAL_ANN_FILE}" data.test.queue_length=0
  data.test.future_length=0 data.test.future_pred_frame_num=0
  data.test.predict_trajectory=False
  data.test.image_size="(${IMAGE_WIDTH},${IMAGE_HEIGHT})"
)

# nvidia-smi is sampled once per second until both single-GPU tests finish.
nvidia-smi --loop=1 --query-gpu=timestamp,index,memory.used \
  --format=csv,noheader,nounits \
  --id=0,1 > "${OUTPUT_DIR}/gpu_memory_mb.csv" &
monitor_pid=$!
cleanup_monitor() {
  kill "${monitor_pid}" 2>/dev/null || true
  wait "${monitor_pid}" 2>/dev/null || true
}
trap cleanup_monitor EXIT

PYTHONPATH="$(pwd):${PYTHONPATH:-}" CUDA_VISIBLE_DEVICES=0 \
  "${PYTHON_BIN}" tools/test.py "${CONFIG}" "${BASE_CHECKPOINT}" \
  "${common_args[@]}" \
  model.pts_bbox_head.transformer.encoder.use_nearfar_bev=False \
  model.pts_bbox_head.transformer.encoder.nearfar_near_ratio=0.6 \
  model.pts_bbox_head.transformer.encoder.nearfar_far_stride=2 \
  > "${OUTPUT_DIR}/base_gpu0.log" 2>&1 &
base_pid=$!

PYTHONPATH="$(pwd):${PYTHONPATH:-}" CUDA_VISIBLE_DEVICES=1 \
  "${PYTHON_BIN}" tools/test.py "${CONFIG}" "${NEARFAR_CHECKPOINT}" \
  "${common_args[@]}" \
  model.pts_bbox_head.transformer.encoder.use_nearfar_bev=True \
  model.pts_bbox_head.transformer.encoder.nearfar_near_ratio=0.6 \
  model.pts_bbox_head.transformer.encoder.nearfar_far_stride=2 \
  > "${OUTPUT_DIR}/nearfar_gpu1.log" 2>&1 &
nearfar_pid=$!

wait "${base_pid}"
wait "${nearfar_pid}"
cleanup_monitor
trap - EXIT

echo "Completed inference benchmark. Logs: ${OUTPUT_DIR}"
