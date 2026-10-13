#!/usr/bin/env bash
# FarmSim front3: 0.2 m Agri-AMoE crop-focused training ablations.
set -euo pipefail

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1,2,3}"
NUM_GPUS="${NUM_GPUS:-3}"
DATA_ROOT="${DATA_ROOT:-/data/HL/SimData-Occ/SimData}"
BATCH_SIZE="${BATCH_SIZE:-3}"
TOTAL_BATCH_SIZE="${TOTAL_BATCH_SIZE:-27}"
IMAGE_WIDTH="${IMAGE_WIDTH:-512}"
IMAGE_HEIGHT="${IMAGE_HEIGHT:-288}"
EPOCHS="${EPOCHS:-15}"
USE_FP16="${USE_FP16:-1}"

PYTHON_BIN="${PYTHON_BIN:-/home/HL/.conda/envs/dow2/bin/python}"
CONFIG="${CONFIG:-projects/configs/farmsim/farmsim_occ_front3.py}"
PRETRAINED_FROM="${PRETRAINED_FROM:-$(pwd)/pretrained/r101_dcn_fcos3d_pretrain.pth}"
TRAIN_ANN_FILE="${TRAIN_ANN_FILE:-data/farmsim/splits/train.json}"
VAL_ANN_FILE="${VAL_ANN_FILE:-}"
WORKERS_PER_GPU="${WORKERS_PER_GPU:-4}"
CROP_STATS_FILE="${CROP_STATS_FILE:-work_dirs/farmsim_front3_train_crop_stats.json}"
# Select a comma-separated subset from amoe,base,cropsample.
FARMSIM_TASKS="${FARMSIM_TASKS:-cropsample}"

# Agri-AMoE settings from the earlier 96-channel ablation.
AGRI_AMOE_CHANNELS=96
AGRI_AMOE_USE_GRADIENT_ENERGY=1
AGRI_AMOE_USE_SALIENCY=1
AGRI_AMOE_GATE_TEMPERATURE=1.0
CROP_OVERSAMPLE_RATIO="${CROP_OVERSAMPLE_RATIO:-0.15}"
CROP_MIN_VOXELS="${CROP_MIN_VOXELS:-1000}"
SEED="${SEED:-0}"

# Export final-checkpoint predictions/PNGs using the existing preview workflow.
AUTO_SAVE_VISUALIZATION="${AUTO_SAVE_VISUALIZATION:-1}"
PREDICTION_COUNT="${PREDICTION_COUNT:-100}"
SAVE_SAMPLING="${SAVE_SAMPLING:-per-sequence}"
PREVIEW_BATCH_SIZE="${PREVIEW_BATCH_SIZE:-1}"
PREVIEW_GPU="${PREVIEW_GPU:-${CUDA_VISIBLE_DEVICES%%,*}}"
GT_CACHE_ROOT="${GT_CACHE_ROOT:-work_dirs/_shared_gt_previews}"

if [[ "${AUTO_SAVE_VISUALIZATION}" != 0 && "${AUTO_SAVE_VISUALIZATION}" != 1 ]]; then
  echo "AUTO_SAVE_VISUALIZATION must be 0 or 1" >&2
  exit 1
fi
if [[ "${AUTO_SAVE_VISUALIZATION}" == 1 ]]; then
  if [[ ! "${PREDICTION_COUNT}" =~ ^[1-9][0-9]*$ ||
        ! "${PREVIEW_BATCH_SIZE}" =~ ^[1-9][0-9]*$ ]]; then
    echo "PREDICTION_COUNT and PREVIEW_BATCH_SIZE must be positive integers" >&2
    exit 1
  fi
  if [[ "${SAVE_SAMPLING}" != per-sequence && "${SAVE_SAMPLING}" != leading ]]; then
    echo "SAVE_SAMPLING must be per-sequence or leading" >&2
    exit 1
  fi
fi

IFS=',' read -r -a selected_tasks <<< "${FARMSIM_TASKS}"
for task in "${selected_tasks[@]}"; do
  case "${task}" in
    amoe|base|cropsample) ;;
    *) echo "Unknown FARMSIM_TASKS task: ${task}; choose amoe,base,cropsample" >&2; exit 1 ;;
  esac
done

[[ -f "${PRETRAINED_FROM}" ]] || {
  echo "Missing pretrained checkpoint: ${PRETRAINED_FROM}" >&2
  exit 1
}

run_experiment() {
  local name="$1"
  local use_agri_amoe="$2"
  shift 2
  local work_dir="work_dirs/${name}_ep${EPOCHS}_$(date +%Y%m%d_%H%M%S)"
  local val_args=()
  if [[ -n "${VAL_ANN_FILE}" ]]; then
    val_args=(--val-ann-file "${VAL_ANN_FILE}")
  fi

  PYTHONPATH="$(pwd):${PYTHONPATH:-}" CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}" \
    "${PYTHON_BIN}" -m torch.distributed.run --standalone \
    --nproc_per_node="${NUM_GPUS}" tools/train.py "${CONFIG}" \
    --launcher pytorch --deterministic --seed "${SEED}" --work-dir "${work_dir}" \
    --load-from "${PRETRAINED_FROM}" \
    --num-gpus "${NUM_GPUS}" --data-root "${DATA_ROOT}" \
    --train-ann-file "${TRAIN_ANN_FILE}" \
    "${val_args[@]}" --workers-per-gpu "${WORKERS_PER_GPU}" \
    --batch-size "${BATCH_SIZE}" --total-batch-size "${TOTAL_BATCH_SIZE}" \
    --image-width "${IMAGE_WIDTH}" --image-height "${IMAGE_HEIGHT}" \
    --use-fp16 "${USE_FP16}" \
    --use-crop-gap-refinement 0 --use-selective-c2f 0 \
    --use-gap-residual-refiner 0 --use-dual-hardness-refinement 0 \
    --use-gvad-attention 0 --use-gvadv2-attention 0 \
    --use-agri-amoe-decoder "${use_agri_amoe}" \
    --agri-amoe-channels "${AGRI_AMOE_CHANNELS}" \
    --agri-amoe-use-gradient-energy "${AGRI_AMOE_USE_GRADIENT_ENERGY}" \
    --agri-amoe-use-saliency "${AGRI_AMOE_USE_SALIENCY}" \
    --agri-amoe-gate-temperature "${AGRI_AMOE_GATE_TEMPERATURE}" \
    --use-nearfar-bev 0 \
    --history-frames 0 --predict-future-occ 0 --future-occ-steps 0 \
    --predict-future-traj 0 --future-traj-steps 6 --epochs "${EPOCHS}" \
    "$@"

  if [[ "${AUTO_SAVE_VISUALIZATION}" == 1 ]]; then
    local trained_config="${work_dir}/$(basename "${CONFIG}")"
    local checkpoint="${work_dir}/epoch_${EPOCHS}.pth"
    local prediction_dir="${work_dir}/prediction"
    local visualization_dir="${work_dir}/visualization"
    local evaluation_log="${work_dir}/evaluation_epoch_${EPOCHS}.log"
    if [[ ! -f "${checkpoint}" || ! -f "${trained_config}" ]]; then
      echo "Cannot export previews: missing final checkpoint or dumped config in ${work_dir}" >&2
      return 1
    fi
    echo "Exporting final-checkpoint previews on GPU ${PREVIEW_GPU}; log: ${evaluation_log}"
    # The dumped config preserves the actual validation split, image size,
    # history length and model switches. Do not rebuild from the base config.
    if ! PYTHONPATH="$(pwd):${PYTHONPATH:-}" CUDA_VISIBLE_DEVICES="${PREVIEW_GPU}" \
      "${PYTHON_BIN}" tools/test.py "${trained_config}" "${checkpoint}" \
      --launcher none --eval occ --deterministic --seed "${SEED}" \
      --batch-size "${PREVIEW_BATCH_SIZE}" \
      --save-predictions "${prediction_dir}" \
      --save-prediction-count "${PREDICTION_COUNT}" \
      --save-prediction-sampling "${SAVE_SAMPLING}" \
      --out "${work_dir}/evaluation_epoch_${EPOCHS}_metrics.pkl" \
      >"${evaluation_log}" 2>&1; then
      echo "Final-checkpoint prediction failed; inspect ${evaluation_log}" >&2
      return 1
    fi
    "${PYTHON_BIN}" tools/render_occ_prediction_previews.py \
      "${prediction_dir}" "${visualization_dir}" \
      --shared-gt-dir "${GT_CACHE_ROOT}/farmsim" \
      --count "${PREDICTION_COUNT}" --max-points 30000 \
      --elev 26 --azim -135 --style auto
    echo "Saved previews: ${prediction_dir}, ${visualization_dir}"
  fi
}

should_run() {
  [[ ",${FARMSIM_TASKS}," == *",$1,"* ]]
}

if should_run amoe; then
  run_experiment front3_base_agri_amoe96_nohis 1
fi
if should_run base; then
  run_experiment front3_base_nohis 0
fi
if should_run cropsample; then
  "${PYTHON_BIN}" tools/build_farmsim_crop_stats.py \
    --manifest "${TRAIN_ANN_FILE}" --data-root "${DATA_ROOT}" \
    --output "${CROP_STATS_FILE}"
fi
if should_run cropsample; then
  run_experiment "front3_base_agri_amoe96_cropsample${CROP_OVERSAMPLE_RATIO}_min${CROP_MIN_VOXELS}_seed${SEED}_nohis" 1 \
    --use-crop-positive-sampling 1 --crop-stats-file "${CROP_STATS_FILE}" \
    --crop-oversample-ratio "${CROP_OVERSAMPLE_RATIO}" \
    --crop-oversample-min-voxels "${CROP_MIN_VOXELS}"
fi
