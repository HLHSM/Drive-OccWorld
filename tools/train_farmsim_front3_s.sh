#!/usr/bin/env bash
# FarmSim front3-s: exact x=[0,15] m, y=[-7.5,7.5] m, z=[-2,3] m.
# Independent 0.2 m / 75x75x25 occupancy training ablations.
set -euo pipefail

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-2}"
IFS=',' read -r -a visible_devices <<< "${CUDA_VISIBLE_DEVICES}"
NUM_GPUS="${NUM_GPUS:-${#visible_devices[@]}}"
DATA_ROOT="${DATA_ROOT:-/data/HL/SimData-Occ/SimData}"

BATCH_SIZE="${BATCH_SIZE:-4}"
TOTAL_BATCH_SIZE="${TOTAL_BATCH_SIZE:-$((BATCH_SIZE * NUM_GPUS * 6))}" #24
IMAGE_WIDTH="${IMAGE_WIDTH:-512}"
IMAGE_HEIGHT="${IMAGE_HEIGHT:-288}"
EPOCHS="${EPOCHS:-15}"
USE_FP16="${USE_FP16:-1}"
SEED="${SEED:-0}"

PYTHON_BIN="${PYTHON_BIN:-/home/HL/.conda/envs/dow2/bin/python}"
CONFIG="${CONFIG:-projects/configs/farmsim/farmsim_occ_front3_s.py}"
PRETRAINED_FROM="${PRETRAINED_FROM:-$(pwd)/pretrained/r101_dcn_fcos3d_pretrain.pth}"
TRAIN_ANN_FILE="${TRAIN_ANN_FILE:-data/farmsim/splits/train.json}"
VAL_ANN_FILE="${VAL_ANN_FILE:-}"


WORKERS_PER_GPU="${WORKERS_PER_GPU:-4}"
CROP_STATS_FILE="${CROP_STATS_FILE:-work_dirs/farmsim_front3_s_train_crop_stats.json}"
CROP_OVERSAMPLE_RATIO="${CROP_OVERSAMPLE_RATIO:-0.15}"
CROP_MIN_VOXELS="${CROP_MIN_VOXELS:-1000}"

# Select any subset: base,amoe,cropsample,amoe_cropsample,gvad_amoe_cropsample.
FARMSIM_TASKS="${FARMSIM_TASKS:-base,amoe,cropsample,amoe_cropsample,gvad_amoe_cropsample}"

AGRI_AMOE_CHANNELS=96
AGRI_AMOE_USE_GRADIENT_ENERGY=1
AGRI_AMOE_USE_SALIENCY=1
AGRI_AMOE_GATE_TEMPERATURE=1.0

[[ -f "${PRETRAINED_FROM}" ]] || {
  echo "Missing pretrained checkpoint: ${PRETRAINED_FROM}" >&2
  exit 1
}
[[ -f "${CONFIG}" ]] || {
  echo "Missing config: ${CONFIG}" >&2
  exit 1
}

IFS=',' read -r -a selected_tasks <<< "${FARMSIM_TASKS}"
for task in "${selected_tasks[@]}"; do
  case "${task}" in
    base|amoe|cropsample|amoe_cropsample|gvad_amoe_cropsample) ;;
    *) echo "Unknown task: ${task}; choose base,amoe,cropsample,amoe_cropsample,gvad_amoe_cropsample" >&2; exit 1 ;;
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
  # Record the effective batch in the directory name before launching. Do not
  # rename or move this directory while training: MMCV keeps its original path
  # for JSON logs and checkpoints throughout the run.
  local work_dir="work_dirs/${name}_b${TOTAL_BATCH_SIZE}_ep${EPOCHS}_$(date +%Y%m%d_%H%M%S)"
  local val_args=()
  if [[ -n "${VAL_ANN_FILE}" ]]; then
    val_args=(--val-ann-file "${VAL_ANN_FILE}")
  fi

  echo "Starting ${name}: ROI x=[0,15], y=[-7.5,7.5], grid=75x75x25; effective batch=${TOTAL_BATCH_SIZE}; output=${work_dir}"
  PYTHONPATH="$(pwd):${PYTHONPATH:-}" CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}" \
    "${PYTHON_BIN}" -m torch.distributed.run --standalone \
    --nproc_per_node="${NUM_GPUS}" tools/train.py "${CONFIG}" \
    --launcher pytorch --deterministic --seed "${SEED}" --work-dir "${work_dir}" \
    --load-from "${PRETRAINED_FROM}" \
    --num-gpus "${NUM_GPUS}" --data-root "${DATA_ROOT}" \
    --train-ann-file "${TRAIN_ANN_FILE}" "${val_args[@]}" \
    --workers-per-gpu "${WORKERS_PER_GPU}" \
    --batch-size "${BATCH_SIZE}" --total-batch-size "${TOTAL_BATCH_SIZE}" \
    --image-width "${IMAGE_WIDTH}" --image-height "${IMAGE_HEIGHT}" \
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
    "$@"
}

# if should_run base; then
#   run_experiment front3_s_base_nohis 0 0
# fi
# if should_run amoe; then
#   run_experiment front3_s_base_agri_amoe96_nohis 1 0
# fi

if should_run cropsample || should_run amoe_cropsample || should_run gvad_amoe_cropsample; then
  "${PYTHON_BIN}" tools/build_farmsim_crop_stats.py \
    --manifest "${TRAIN_ANN_FILE}" --data-root "${DATA_ROOT}" \
    --target-point-cloud-range 0 -7.5 -2 15 7.5 3 \
    --output "${CROP_STATS_FILE}"
fi

# CROP_MIN_VOXELS=563

# if should_run amoe_cropsample; then
#   run_experiment "front3_s_base_agri_amoe96_cropsample${CROP_OVERSAMPLE_RATIO}_min${CROP_MIN_VOXELS}_seed${SEED}_nohis" 1 0 \
#     --use-crop-positive-sampling 1 --crop-stats-file "${CROP_STATS_FILE}" \
#     --crop-oversample-ratio "${CROP_OVERSAMPLE_RATIO}" \
#     --crop-oversample-min-voxels "${CROP_MIN_VOXELS}"
# fi

# CROP_MIN_VOXELS=563

# if should_run amoe_cropsample; then
#   run_experiment "front3_s_base_agri_amoe96_cropsample${CROP_OVERSAMPLE_RATIO}_min${CROP_MIN_VOXELS}_nohis" 1 0 \
#     --use-crop-positive-sampling 1 --crop-stats-file "${CROP_STATS_FILE}" \
#     --crop-oversample-ratio "${CROP_OVERSAMPLE_RATIO}" \
#     --crop-oversample-min-voxels "${CROP_MIN_VOXELS}"
# fi

CROP_MIN_VOXELS=500
CROP_OVERSAMPLE_RATIO=0.1
# if should_run amoe_cropsample; then
#   run_experiment "front3_s_base_agri_amoe96_cropsample${CROP_OVERSAMPLE_RATIO}_min${CROP_MIN_VOXELS}_nohis" 1 0 \
#     --use-crop-positive-sampling 1 --crop-stats-file "${CROP_STATS_FILE}" \
#     --crop-oversample-ratio "${CROP_OVERSAMPLE_RATIO}" \
#     --crop-oversample-min-voxels "${CROP_MIN_VOXELS}"
# fi

if should_run gvad_amoe_cropsample; then
  run_experiment "front3_s_base_gvad_agri_amoe96_cropsample${CROP_OVERSAMPLE_RATIO}_min${CROP_MIN_VOXELS}_nohis" 1 1 \
    --use-crop-positive-sampling 1 --crop-stats-file "${CROP_STATS_FILE}" \
    --crop-oversample-ratio "${CROP_OVERSAMPLE_RATIO}" \
    --crop-oversample-min-voxels "${CROP_MIN_VOXELS}"
fi

# if should_run cropsample; then
#   run_experiment "front3_s_base_cropsample${CROP_OVERSAMPLE_RATIO}_min${CROP_MIN_VOXELS}_seed${SEED}_nohis" 0 0 \
#     --use-crop-positive-sampling 1 --crop-stats-file "${CROP_STATS_FILE}" \
#     --crop-oversample-ratio "${CROP_OVERSAMPLE_RATIO}" \
#     --crop-oversample-min-voxels "${CROP_MIN_VOXELS}"
# fi
