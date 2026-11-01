#!/usr/bin/env bash
# FarmSim front3-s: ROI [0,-7.5,-2,15,7.5,3], 75x75x25, current frame only.
# Run independent ablations sequentially on the same GPUs. No CropSample by default.
#
# Examples (run from the repository root):
#   DRY_RUN=1 bash tools/train_farmsim_front3_s.sh
#   FARMSIM_TASKS=base_no_tsa bash tools/train_farmsim_front3_s.sh
#   FARMSIM_TASKS=amoe,amoe_no_channel,amoe_no_saliency,amoe_no_energy bash tools/train_farmsim_front3_s.sh
#
# Task names match the paper's ablation rows:
#   base_no_tsa (alias: base)   MLP head; remove TSA and its following LayerNorm
#   base_tsa                   MLP head with TSA
#   amoe                       full Agri-AMoE with TSA, no CropSample
#   amoe_no_channel            bypass channel attention, retain spatial saliency
#   amoe_planar                three identical local 1x3x3 expert architectures
#   amoe_no_xy                 remove local planar expert (index 0)
#   amoe_no_z                  remove vertical expert (index 1)
#   amoe_no_context            remove dilated context expert (index 2)
#   amoe_mean                  fixed uniform expert weights
#   amoe_global                learned weights shared by all samples and voxels
#   amoe_no_saliency           zero spatial routing signal, retain channel attention
#   amoe_no_energy             zero gradient-energy routing signal
#   amoe_cropsample            optional original full model with CropSample
#   all                        all ablations above, including base_tsa, except CropSample
#
# CAMERA_GEOMETRY defaults to the existing grid_world setup. To reproduce the
# paper's older legacy-projection runs, set CAMERA_GEOMETRY=legacy for ALL controls
# and ablations; compare only runs using the same projection mode.
set -euo pipefail

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"
IFS=',' read -r -a visible_devices <<< "${CUDA_VISIBLE_DEVICES}"
NUM_GPUS="${NUM_GPUS:-${#visible_devices[@]}}"
DATA_ROOT="${DATA_ROOT:-/data/HL/SimData-Occ/SimData}"
BATCH_SIZE="${BATCH_SIZE:-4}"
TOTAL_BATCH_SIZE="${TOTAL_BATCH_SIZE:-24}"
IMAGE_WIDTH="${IMAGE_WIDTH:-512}"
IMAGE_HEIGHT="${IMAGE_HEIGHT:-288}"
CAMERA_GEOMETRY="${CAMERA_GEOMETRY:-grid_world}"
EPOCHS="${EPOCHS:-15}"
USE_FP16="${USE_FP16:-1}"
SEED="${SEED:-0}"
DRY_RUN="${DRY_RUN:-0}"
LOG_INTERVAL="${LOG_INTERVAL:-20}"
AGRI_AMOE_CHANNELS="${AGRI_AMOE_CHANNELS:-96}"

PYTHON_BIN="${PYTHON_BIN:-/home/HL/.conda/envs/dow2/bin/python}"
CONFIG="${CONFIG:-projects/configs/farmsim/farmsim_occ_front3_s.py}"
PRETRAINED_FROM="${PRETRAINED_FROM:-$(pwd)/pretrained/r101_dcn_fcos3d_pretrain.pth}"
TRAIN_ANN_FILE="${TRAIN_ANN_FILE:-data/farmsim/splits/train.json}"
VAL_ANN_FILE="${VAL_ANN_FILE:-data/farmsim/splits/val.json}"
WORKERS_PER_GPU="${WORKERS_PER_GPU:-4}"
CROP_STATS_FILE="${CROP_STATS_FILE:-work_dirs/farmsim_front3_s_train_crop_stats.json}"
CROP_OVERSAMPLE_RATIO="${CROP_OVERSAMPLE_RATIO:-0.1}"
CROP_MIN_VOXELS="${CROP_MIN_VOXELS:-500}"

DEFAULT_TASKS="base_no_tsa,amoe,amoe_no_channel,amoe_planar,amoe_no_xy,amoe_no_z,amoe_no_context,amoe_mean,amoe_global,amoe_no_saliency,amoe_no_energy"
FARMSIM_TASKS="${FARMSIM_TASKS:-${DEFAULT_TASKS}}"
if [[ "${FARMSIM_TASKS}" == all ]]; then
  FARMSIM_TASKS="base_tsa,${DEFAULT_TASKS}"
fi

case "${CAMERA_GEOMETRY}" in
  legacy|grid_world) ;;
  *) echo "Invalid CAMERA_GEOMETRY=${CAMERA_GEOMETRY}; choose grid_world or legacy" >&2; exit 1 ;;
esac
case "${DRY_RUN}" in
  0|1) ;;
  *) echo "DRY_RUN must be 0 or 1" >&2; exit 1 ;;
esac
for value in "${NUM_GPUS}" "${BATCH_SIZE}" "${TOTAL_BATCH_SIZE}"; do
  [[ "${value}" =~ ^[1-9][0-9]*$ ]] || {
    echo "NUM_GPUS, BATCH_SIZE and TOTAL_BATCH_SIZE must be positive integers" >&2
    exit 1
  }
done
if (( NUM_GPUS != ${#visible_devices[@]} )); then
  echo "NUM_GPUS must match the number of CUDA_VISIBLE_DEVICES entries" >&2
  exit 1
fi
if (( TOTAL_BATCH_SIZE % (BATCH_SIZE * NUM_GPUS) != 0 )); then
  echo "TOTAL_BATCH_SIZE must be divisible by BATCH_SIZE * NUM_GPUS" >&2
  exit 1
fi
[[ -f "${CONFIG}" ]] || { echo "Missing config: ${CONFIG}" >&2; exit 1; }

IFS=',' read -r -a selected_tasks <<< "${FARMSIM_TASKS}"
for task in "${selected_tasks[@]}"; do
  case "${task}" in
    base|base_no_tsa|base_tsa|amoe|amoe_no_channel|amoe_planar|amoe_no_xy|amoe_no_z|amoe_no_context|amoe_mean|amoe_global|amoe_no_saliency|amoe_no_energy|amoe_cropsample) ;;
    *) echo "Unknown task: ${task}; see task names at the top of this script" >&2; exit 1 ;;
  esac
done

run_experiment() {
  local task="$1"
  local use_agri_amoe=1 disable_tsa=0 crop_sampling=0
  local channel_attention=True spatial_saliency=True gradient_energy=1
  local expert_layout=anisotropic active_experts='[0,1,2]' fusion=voxel
  case "${task}" in
    base|base_no_tsa) use_agri_amoe=0; disable_tsa=1 ;;
    base_tsa) use_agri_amoe=0 ;;
    amoe_no_channel) channel_attention=False ;;
    amoe_planar) expert_layout=planar ;;
    amoe_no_xy) active_experts='[1,2]' ;;
    amoe_no_z) active_experts='[0,2]' ;;
    amoe_no_context) active_experts='[0,1]' ;;
    amoe_mean) fusion=mean ;;
    amoe_global) fusion=global ;;
    amoe_no_saliency) spatial_saliency=False ;;
    amoe_no_energy) gradient_energy=0 ;;
    amoe_cropsample) crop_sampling=1 ;;
  esac
  local name="front3_s_${task}_amoe${AGRI_AMOE_CHANNELS}_nohis"
  if [[ "${use_agri_amoe}" == 0 ]]; then
    name="front3_s_${task}_nohis"
  fi
  local work_dir="work_dirs/${name}_${CAMERA_GEOMETRY}_seed${SEED}_b${TOTAL_BATCH_SIZE}_ep${EPOCHS}_$(date +%Y%m%d_%H%M%S)"
  local val_args=() crop_args=()
  if [[ -n "${VAL_ANN_FILE}" ]]; then
    val_args=(--val-ann-file "${VAL_ANN_FILE}")
  fi
  if [[ "${crop_sampling}" == 1 ]]; then
    crop_args=(--crop-stats-file "${CROP_STATS_FILE}"
               --crop-oversample-ratio "${CROP_OVERSAMPLE_RATIO}"
               --crop-oversample-min-voxels "${CROP_MIN_VOXELS}")
  fi
  # Explicitly reset every ablation field, including the uniform sampler, so
  # a supplied saved config cannot silently carry a previous ablation forward.
  local sampler_type=DistributedGroupSampler
  if [[ "${crop_sampling}" == 1 ]]; then
    sampler_type=CropAwareDistributedGroupSampler
  fi
  local command=(
    "${PYTHON_BIN}" -m torch.distributed.run --standalone
    --nproc_per_node="${NUM_GPUS}" --log-dir "${work_dir}/torchrun" --tee 3
    tools/train.py "${CONFIG}"
    --launcher pytorch --deterministic --seed "${SEED}" --work-dir "${work_dir}"
    --load-from "${PRETRAINED_FROM}"
    --num-gpus "${NUM_GPUS}" --data-root "${DATA_ROOT}"
    --train-ann-file "${TRAIN_ANN_FILE}" "${val_args[@]}"
    --workers-per-gpu "${WORKERS_PER_GPU}"
    --batch-size "${BATCH_SIZE}" --total-batch-size "${TOTAL_BATCH_SIZE}"
    --image-width "${IMAGE_WIDTH}" --image-height "${IMAGE_HEIGHT}"
    --camera-geometry "${CAMERA_GEOMETRY}"
    --use-fp16 "${USE_FP16}" --epochs "${EPOCHS}"
    --history-frames 0 --predict-future-occ 0 --future-occ-steps 0
    --predict-future-traj 0 --future-traj-steps 6
    --disable-temporal-self-attention "${disable_tsa}"
    --use-r50-image-encoder 0 --use-gvad-attention 0 --use-gvadv2-attention 0
    --use-directional-decay-retention 0 --use-nearfar-bev 0
    --use-crop-gap-refinement 0 --use-selective-c2f 0
    --use-gap-residual-refiner 0 --use-dual-hardness-refinement 0
    --use-agri-amoe-decoder "${use_agri_amoe}"
    --agri-amoe-channels "${AGRI_AMOE_CHANNELS}"
    --agri-amoe-use-gradient-energy "${gradient_energy}"
    --agri-amoe-use-saliency 1 --agri-amoe-gate-temperature 1.0
    --use-crop-positive-sampling "${crop_sampling}" "${crop_args[@]}"
    --cfg-options log_config.interval="${LOG_INTERVAL}" evaluation.interval=1
    "data.shuffler_sampler._delete_=True"
    "data.shuffler_sampler.type=${sampler_type}"
    "model.future_pred_head.agri_amoe_use_channel_attention=${channel_attention}"
    "model.future_pred_head.agri_amoe_use_spatial_saliency=${spatial_saliency}"
    "model.future_pred_head.agri_amoe_expert_layout=${expert_layout}"
    "model.future_pred_head.agri_amoe_active_experts=${active_experts}"
    "model.future_pred_head.agri_amoe_fusion=${fusion}"
  )
  local stats_command=(
    "${PYTHON_BIN}" tools/build_farmsim_crop_stats.py
    --manifest "${TRAIN_ANN_FILE}" --data-root "${DATA_ROOT}"
    --target-point-cloud-range 0 -7.5 -2 15 7.5 3 --output "${CROP_STATS_FILE}"
  )
  echo "Task=${task}; TSA=$((1-disable_tsa)); CropSample=${crop_sampling}; camera_geometry=${CAMERA_GEOMETRY}; effective_batch=${TOTAL_BATCH_SIZE}; output=${work_dir}"
  if [[ "${DRY_RUN}" == 1 ]]; then
    if [[ "${crop_sampling}" == 1 ]]; then
      printf '%q ' "${stats_command[@]}"
      printf '\n'
    fi
    printf '%q ' env "PYTHONPATH=$(pwd):${PYTHONPATH:-}" \
      "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}" PYTHONUNBUFFERED=1 FARMSIM_TRACE_SIGTERM=1 "${command[@]}"
    printf '\n'
    return
  fi
  [[ -f "${PRETRAINED_FROM}" ]] || {
    echo "Missing pretrained checkpoint: ${PRETRAINED_FROM}" >&2; exit 1
  }
  if [[ "${crop_sampling}" == 1 ]]; then
    "${stats_command[@]}"
  fi
  mkdir -p "${work_dir}/torchrun"
  local launcher_log="${work_dir}/launcher.log" exit_status=0
  printf 'launcher_shell_pid=%s parent_pid=%s CUDA_VISIBLE_DEVICES=%s\n' \
    "$$" "${PPID}" "${CUDA_VISIBLE_DEVICES}" | tee -a "${launcher_log}"
  printf '%q ' "${command[@]}" >> "${launcher_log}"
  printf '\n' >> "${launcher_log}"
  PYTHONPATH="$(pwd):${PYTHONPATH:-}" CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}" \
    PYTHONUNBUFFERED=1 FARMSIM_TRACE_SIGTERM=1 \
    "${command[@]}" 2>&1 | tee -a "${launcher_log}" || exit_status=$?
  printf '%s\n' "${exit_status}" > "${work_dir}/launcher.exitcode"
  if (( exit_status != 0 )); then
    echo "Training exited with status ${exit_status}; inspect ${launcher_log} and ${work_dir}/torchrun/*/attempt_*/<rank>/." >&2
    return "${exit_status}"
  fi
}

for task in "${selected_tasks[@]}"; do
  run_experiment "${task}"
done
