#!/usr/bin/env bash
# FarmSim front3-s: exact x=[0,15] m, y=[-7.5,7.5] m, z=[-2,3] m.
# Independent 0.2 m / 75x75x25 occupancy training ablations.
set -euo pipefail

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1,2,3}"
IFS=',' read -r -a visible_devices <<< "${CUDA_VISIBLE_DEVICES}"
NUM_GPUS="${NUM_GPUS:-${#visible_devices[@]}}"
DATA_ROOT="${DATA_ROOT:-/data/HL/SimData-Occ/SimData}"

BATCH_SIZE="${BATCH_SIZE:-4}"
TOTAL_BATCH_SIZE="${TOTAL_BATCH_SIZE:-$((BATCH_SIZE * NUM_GPUS * 2))}" #24
IMAGE_WIDTH="${IMAGE_WIDTH:-512}"
IMAGE_HEIGHT="${IMAGE_HEIGHT:-288}"
# grid_world: measured camera/world/grid transforms; legacy: old mount-only projection.
CAMERA_GEOMETRY="${CAMERA_GEOMETRY:-grid_world}"
EPOCHS="${EPOCHS:-15}"
USE_FP16="${USE_FP16:-1}"
SEED="${SEED:-0}"

PYTHON_BIN="${PYTHON_BIN:-/home/HL/.conda/envs/dow2/bin/python}"
CONFIG="${CONFIG:-projects/configs/farmsim/farmsim_occ_front3_s.py}"
PRETRAINED_FROM="${PRETRAINED_FROM:-$(pwd)/pretrained/r101_dcn_fcos3d_pretrain.pth}"
R50_PRETRAINED_FROM="${R50_PRETRAINED_FROM:-$(pwd)/pretrained/r50_fcos3d_pretrain.pth}"
TRAIN_ANN_FILE="${TRAIN_ANN_FILE:-data/farmsim/splits/train.json}"
VAL_ANN_FILE="${VAL_ANN_FILE:-data/farmsim/splits/val.json}"


WORKERS_PER_GPU="${WORKERS_PER_GPU:-4}"
CROP_STATS_FILE="${CROP_STATS_FILE:-work_dirs/farmsim_front3_s_train_crop_stats.json}"
CROP_OVERSAMPLE_RATIO="${CROP_OVERSAMPLE_RATIO:-0.1}"
CROP_MIN_VOXELS="${CROP_MIN_VOXELS:-500}"

# Default: R101 base + Agri-AMoE + crop sampling, with corrected projection.
# Select kd_control,kd_logits,kd_fpn,kd_both explicitly for distillation runs.
# Use CAMERA_GEOMETRY=legacy with the existing legacy-projection KD checkpoints.
FARMSIM_TASKS="${FARMSIM_TASKS:-amoe_cropsample}"
KD_STUDENT_CHECKPOINT="${KD_STUDENT_CHECKPOINT:-work_dirs/front3_s_base_r50_agri_amoe96_cropsample0.1_min500_nohis_b24_ep15_20261014_200858/epoch_15.pth}"
KD_TEACHER_DIR="${KD_TEACHER_DIR:-work_dirs/front3_s_base_agri_amoe96_cropsample0.1_min500_nohis_b24_ep15_20261012_214738}"
KD_TEACHER_CHECKPOINT="${KD_TEACHER_CHECKPOINT:-${KD_TEACHER_DIR}/epoch_15.pth}"
KD_TEACHER_CONFIG="${KD_TEACHER_CONFIG:-${KD_TEACHER_DIR}/farmsim_occ_front3_s.py}"
KD_EPOCHS="${KD_EPOCHS:-5}"
KD_LR="${KD_LR:-5e-5}"
KD_LR_WARMUP_ITERS="${KD_LR_WARMUP_ITERS:-300}"
KD_WARMUP_EPOCHS="${KD_WARMUP_EPOCHS:-0.5}"
KD_LOGIT_WEIGHT="${KD_LOGIT_WEIGHT:-1.0}"
KD_FPN_WEIGHT="${KD_FPN_WEIGHT:-1.0}"
KD_LOGIT_TEMPERATURE="${KD_LOGIT_TEMPERATURE:-2.0}"
KD_FPN_TEMPERATURE="${KD_FPN_TEMPERATURE:-1.0}"
KD_FP16_INITIAL_SCALE="${KD_FP16_INITIAL_SCALE:-128.0}"
KD_RUN_SUFFIX="${KD_RUN_SUFFIX:-}"
LOG_INTERVAL="${LOG_INTERVAL:-20}"

AGRI_AMOE_CHANNELS=96
AGRI_AMOE_USE_GRADIENT_ENERGY=1
AGRI_AMOE_USE_SALIENCY=1
AGRI_AMOE_GATE_TEMPERATURE=1.0

case "${CAMERA_GEOMETRY}" in
  legacy|grid_world) ;;
  *) echo "Invalid CAMERA_GEOMETRY=${CAMERA_GEOMETRY}; choose grid_world or legacy" >&2; exit 1 ;;
esac

[[ -f "${CONFIG}" ]] || {
  echo "Missing config: ${CONFIG}" >&2
  exit 1
}

IFS=',' read -r -a selected_tasks <<< "${FARMSIM_TASKS}"
for task in "${selected_tasks[@]}"; do
  case "${task}" in
    base|amoe|cropsample|amoe_cropsample|gvad_amoe_cropsample|r50_amoe_cropsample|kd_control|kd_logits|kd_fpn|kd_both) ;;
    *) echo "Unknown task: ${task}; choose base,amoe,cropsample,amoe_cropsample,gvad_amoe_cropsample,r50_amoe_cropsample,kd_control,kd_logits,kd_fpn,kd_both" >&2; exit 1 ;;
  esac
done

should_run() {
  [[ ",${FARMSIM_TASKS}," == *",$1,"* ]]
}

run_experiment() {
  local name="$1"
  local use_agri_amoe="$2"
  local use_gvad_attention="$3"
  shift 3
  local pretrained_from="${EXPERIMENT_PRETRAINED_FROM:-${PRETRAINED_FROM}}"
  [[ -f "${pretrained_from}" ]] || {
    echo "Missing pretrained checkpoint: ${pretrained_from}" >&2
    exit 1
  }
  # Record the effective batch in the directory name before launching. Do not
  # rename or move this directory while training: MMCV keeps its original path
  # for JSON logs and checkpoints throughout the run.
  local work_dir="work_dirs/${name}_${CAMERA_GEOMETRY}_b${TOTAL_BATCH_SIZE}_ep${EPOCHS}_$(date +%Y%m%d_%H%M%S)"
  local val_args=()
  if [[ -n "${VAL_ANN_FILE}" ]]; then
    val_args=(--val-ann-file "${VAL_ANN_FILE}")
  fi

  mkdir -p "${work_dir}/torchrun"
  local launcher_log="${work_dir}/launcher.log"
  local exit_status=0
  echo "Starting ${name}: ROI x=[0,15], y=[-7.5,7.5], grid=75x75x25; camera_geometry=${CAMERA_GEOMETRY}; effective batch=${TOTAL_BATCH_SIZE}; output=${work_dir}"
  printf 'launcher_shell_pid=%s parent_pid=%s CUDA_VISIBLE_DEVICES=%s\n' \
    "$$" "${PPID}" "${CUDA_VISIBLE_DEVICES}" | tee -a "${launcher_log}"
  PYTHONPATH="$(pwd):${PYTHONPATH:-}" CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}" \
    PYTHONUNBUFFERED=1 FARMSIM_TRACE_SIGTERM=1 \
    "${PYTHON_BIN}" -m torch.distributed.run --standalone \
    --nproc_per_node="${NUM_GPUS}" --log-dir "${work_dir}/torchrun" --tee 3 \
    tools/train.py "${CONFIG}" \
    --launcher pytorch --deterministic --seed "${SEED}" --work-dir "${work_dir}" \
    --load-from "${pretrained_from}" \
    --num-gpus "${NUM_GPUS}" --data-root "${DATA_ROOT}" \
    --train-ann-file "${TRAIN_ANN_FILE}" "${val_args[@]}" \
    --workers-per-gpu "${WORKERS_PER_GPU}" \
    --batch-size "${BATCH_SIZE}" --total-batch-size "${TOTAL_BATCH_SIZE}" \
    --image-width "${IMAGE_WIDTH}" --image-height "${IMAGE_HEIGHT}" \
    --camera-geometry "${CAMERA_GEOMETRY}" \
    --use-fp16 "${USE_FP16}" --epochs "${EPOCHS}" \
    --history-frames 0 --predict-future-occ 0 --future-occ-steps 0 \
    --predict-future-traj 0 --future-traj-steps 6 \
    --disable-temporal-self-attention 0 --use-gvad-attention "${use_gvad_attention}" \
    --use-gvadv2-attention 0 --use-nearfar-bev 0 \
    --use-crop-gap-refinement 0 --use-selective-c2f 0 \
    --use-gap-residual-refiner 0 --use-dual-hardness-refinement 0 \
    --use-agri-amoe-decoder "${use_agri_amoe}" \
    --agri-amoe-channels "${AGRI_AMOE_CHANNELS}" \
    --agri-amoe-use-gradient-energy "${AGRI_AMOE_USE_GRADIENT_ENERGY}" \
    --agri-amoe-use-saliency "${AGRI_AMOE_USE_SALIENCY}" \
    --agri-amoe-gate-temperature "${AGRI_AMOE_GATE_TEMPERATURE}" \
    "$@" 2>&1 | tee -a "${launcher_log}" || exit_status=$?
  printf '%s\n' "${exit_status}" > "${work_dir}/launcher.exitcode"
  if (( exit_status != 0 )); then
    echo "Training exited with status ${exit_status}; inspect ${launcher_log} and ${work_dir}/torchrun/*/attempt_*/<rank>/." >&2
    return "${exit_status}"
  fi
}

# if should_run base; then
#   run_experiment front3_s_base_nohis 0 0
# fi
# if should_run amoe; then
#   run_experiment front3_s_base_agri_amoe96_nohis 1 0
# fi

if should_run cropsample || should_run amoe_cropsample || should_run gvad_amoe_cropsample || should_run r50_amoe_cropsample || should_run kd_control || should_run kd_logits || should_run kd_fpn || should_run kd_both; then
  "${PYTHON_BIN}" tools/build_farmsim_crop_stats.py \
    --manifest "${TRAIN_ANN_FILE}" --data-root "${DATA_ROOT}" \
    --target-point-cloud-range 0 -7.5 -2 15 7.5 3 \
    --output "${CROP_STATS_FILE}"
fi

if should_run amoe_cropsample; then
  run_experiment "front3_s_base_agri_amoe96_cropsample${CROP_OVERSAMPLE_RATIO}_min${CROP_MIN_VOXELS}_nohis" 1 0 \
    --use-r50-image-encoder 0 \
    --use-crop-positive-sampling 1 --crop-stats-file "${CROP_STATS_FILE}" \
    --crop-oversample-ratio "${CROP_OVERSAMPLE_RATIO}" \
    --crop-oversample-min-voxels "${CROP_MIN_VOXELS}" \
    --cfg-options log_config.interval="${LOG_INTERVAL}" evaluation.interval=1
fi

# # All four runs load identical R50 epoch-15 weights into a NEW optimizer and
# # a five-epoch LR schedule. They do not resume the exhausted epoch-15 schedule.
# run_kd_experiment() {
#   local label="$1"
#   local logit_weight="$2"
#   local fpn_weight="$3"
#   local EPOCHS="${KD_EPOCHS}"
#   local EXPERIMENT_PRETRAINED_FROM="${KD_STUDENT_CHECKPOINT}"
#   local teacher_args=()
#   local amp_args=()
#   if [[ "${USE_FP16}" == "1" ]]; then
#     amp_args=(--fp16-initial-scale "${KD_FP16_INITIAL_SCALE}")
#   fi
#   if [[ "${label}" != "a_control" ]]; then
#     teacher_args=(--distill-teacher-checkpoint "${KD_TEACHER_CHECKPOINT}"
#                   --distill-teacher-config "${KD_TEACHER_CONFIG}")
#   fi
#   run_experiment "front3_s_r50_kd_${label}${KD_RUN_SUFFIX}_agri_amoe96_cropsample${CROP_OVERSAMPLE_RATIO}_min${CROP_MIN_VOXELS}_nohis" 1 0 \
#     --use-r50-image-encoder 1 \
#     --use-crop-positive-sampling 1 --crop-stats-file "${CROP_STATS_FILE}" \
#     --crop-oversample-ratio "${CROP_OVERSAMPLE_RATIO}" \
#     --crop-oversample-min-voxels "${CROP_MIN_VOXELS}" \
#     "${teacher_args[@]}" \
#     "${amp_args[@]}" \
#     --distill-logit-weight "${logit_weight}" --distill-fpn-weight "${fpn_weight}" \
#     --distill-logit-temperature "${KD_LOGIT_TEMPERATURE}" \
#     --distill-fpn-temperature "${KD_FPN_TEMPERATURE}" \
#     --distill-fpn-levels 0 1 --distill-warmup-epochs "${KD_WARMUP_EPOCHS}" \
#     --cfg-options optimizer.lr="${KD_LR}" \
#       lr_config.warmup_iters="${KD_LR_WARMUP_ITERS}" \
#       log_config.interval="${LOG_INTERVAL}" evaluation.interval=1
# }


# # A-D: matched continuation from R50 epoch 15, using the 0.1/min500 sampler.
# if should_run kd_control; then
#   run_kd_experiment a_control 0 0
# fi
# if should_run kd_logits; then
#   run_kd_experiment b_logits "${KD_LOGIT_WEIGHT}" 0
# fi
# if should_run kd_fpn; then
#   run_kd_experiment c_fpn 0 "${KD_FPN_WEIGHT}"
# fi
# if should_run kd_both; then
#   run_kd_experiment d_both "${KD_LOGIT_WEIGHT}" "${KD_FPN_WEIGHT}"
# fi
