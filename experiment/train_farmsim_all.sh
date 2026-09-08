#!/usr/bin/env bash
# Current-frame FarmSim occupancy baselines.  Runs one selected method, or all
# runnable methods sequentially.  IR-WM is intentionally skipped by default:
# its existing experiment is reused rather than retrained.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
METHOD="${1:-all}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
NUM_GPUS="${NUM_GPUS:-4}"
DATA_ROOT="${DATA_ROOT:-/data/HL/SimData-Occ/SimData}"
EPOCHS="${EPOCHS:-8}"
# Shared knobs.  External baselines default to a global effective batch of 24
# at 512x288.  When BATCH_SIZE is omitted, their launcher
# derives the largest no-accumulation per-GPU batch from TOTAL_BATCH_SIZE.
# FP16 remains opt-in: the current COTR and SurroundOcc Torch2 ports have
# model-specific mixed-precision failures, while SparseOcc has been verified.
BATCH_SIZE="${BATCH_SIZE:-3}"
TOTAL_BATCH_SIZE="${TOTAL_BATCH_SIZE:-24}"
IMAGE_WIDTH="${IMAGE_WIDTH:-512}"
IMAGE_HEIGHT="${IMAGE_HEIGHT:-288}"
USE_FP16="${USE_FP16:-1}"
WORKERS_PER_GPU="${WORKERS_PER_GPU:-4}"

DRIVE_ENV="${DRIVE_ENV:-dow2}"
SURROUNDOCC_ENV="${SURROUNDOCC_ENV:-surroundocc20}"
COTR_ENV="${COTR_ENV:-cotr118}"
SPARSE_ENV="${SPARSE_ENV:-sparseocc}"
# Keep the system CUDA/driver untouched.  Legacy OpenMMLab extensions are
# compiled with this user-local CUDA 11.8 Toolkit to match their CUDA wheels.
CUDA118_HOME="${CUDA118_HOME:-/home/HL/.local/cuda-11.8}"
RUN_EXISTING_IRWM="${RUN_EXISTING_IRWM:-1}"

require_positive_int() {
  local name="$1"
  local value="$2"
  [[ "${value}" =~ ^[1-9][0-9]*$ ]] || {
    echo "${name} must be a positive integer, got '${value}'." >&2
    exit 2
  }
}

prepare_external_runtime() {
  local per_gpu
  local fp16="${USE_FP16}"
  local micro_batch
  local requested_total="${TOTAL_BATCH_SIZE}"

  require_positive_int NUM_GPUS "${NUM_GPUS}"
  require_positive_int TOTAL_BATCH_SIZE "${requested_total}"
  if [[ -n "${BATCH_SIZE}" ]]; then
    per_gpu="${BATCH_SIZE}"
  else
    (( requested_total % NUM_GPUS == 0 )) || {
      echo "TOTAL_BATCH_SIZE=${requested_total} must be divisible by NUM_GPUS=${NUM_GPUS} when BATCH_SIZE is omitted." >&2
      exit 2
    }
    per_gpu="$((requested_total / NUM_GPUS))"
  fi
  require_positive_int BATCH_SIZE "${per_gpu}"
  require_positive_int IMAGE_WIDTH "${IMAGE_WIDTH}"
  require_positive_int IMAGE_HEIGHT "${IMAGE_HEIGHT}"
  require_positive_int WORKERS_PER_GPU "${WORKERS_PER_GPU}"
  [[ "${fp16}" == "0" || "${fp16}" == "1" ]] || {
    echo "USE_FP16 must be 0 or 1, got '${fp16}'." >&2
    exit 2
  }

  micro_batch=$((per_gpu * NUM_GPUS))
  (( requested_total % micro_batch == 0 )) || {
    echo "TOTAL_BATCH_SIZE=${requested_total} must be divisible by BATCH_SIZE*NUM_GPUS=${micro_batch}." >&2
    exit 2
  }

  export FARMSIM_BATCH_SIZE="${per_gpu}"
  # Keep both the requested effective batch and the one-step global batch
  # available to external repositories.  The latter is intentionally distinct
  # from TOTAL_BATCH_SIZE when gradient accumulation is enabled.
  export FARMSIM_TOTAL_BATCH_SIZE="${requested_total}"
  export FARMSIM_MICRO_BATCH_SIZE="${micro_batch}"
  export FARMSIM_WORLD_SIZE="${NUM_GPUS}"
  export FARMSIM_GRAD_ACCUM_STEPS="$((requested_total / micro_batch))"
  export FARMSIM_IMAGE_WIDTH="${IMAGE_WIDTH}"
  export FARMSIM_IMAGE_HEIGHT="${IMAGE_HEIGHT}"
  export FARMSIM_USE_FP16="${fp16}"
  export WORKERS_PER_GPU

  echo "External runtime: per-GPU batch=${per_gpu}, world=${NUM_GPUS}, effective batch=${requested_total}, gradient accumulation=${FARMSIM_GRAD_ACCUM_STEPS}, image=${IMAGE_WIDTH}x${IMAGE_HEIGHT}, fp16=${fp16}, workers/GPU=${WORKERS_PER_GPU}."

}

run_native_driveocc() {
  local batch_size="${BATCH_SIZE:-3}"
  local total_batch_size="${TOTAL_BATCH_SIZE:-24}"
  local use_fp16="${USE_FP16:-1}"
  cd "${ROOT}"
  PYTHONPATH="${ROOT}:${PYTHONPATH:-}" CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}" \
    conda run --no-capture-output -n "${DRIVE_ENV}" python -m torch.distributed.run \
    --standalone --nproc_per_node="${NUM_GPUS}" tools/train.py \
    projects/configs/farmsim/farmsim_occ_front3.py --launcher pytorch \
    --deterministic --work-dir "work_dirs/driveocc_farmsim_front3_ep${EPOCHS}" \
    --load-from "${ROOT}/pretrained/r101_dcn_fcos3d_pretrain.pth" \
    --num-gpus "${NUM_GPUS}" --data-root "${DATA_ROOT}" --batch-size "${batch_size}" \
    --total-batch-size "${total_batch_size}" --image-width "${IMAGE_WIDTH}" --image-height "${IMAGE_HEIGHT}" --use-fp16 "${use_fp16}" \
    --workers-per-gpu "${WORKERS_PER_GPU}" \
    --history-frames 0 --predict-future-occ 0 --future-occ-steps 0 \
    --predict-future-traj 0 --future-traj-steps 6 --epochs "${EPOCHS}"
}

run_irwm() {
  if [[ "${RUN_EXISTING_IRWM}" != "1" ]]; then
    echo "IR-WM reuse: work_dirs/front3_irwm_task_ep5_20260826_120050 (set RUN_EXISTING_IRWM=1 to retrain)."
    return
  fi
  local batch_size="${BATCH_SIZE:-3}"
  local total_batch_size="${TOTAL_BATCH_SIZE:-24}"
  local use_fp16="${USE_FP16:-1}"
  cd "${ROOT}"
  PYTHONPATH="${ROOT}:${PYTHONPATH:-}" CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}" \
    conda run --no-capture-output -n "${DRIVE_ENV}" python -m torch.distributed.run \
    --standalone --nproc_per_node="${NUM_GPUS}" tools/train.py \
    projects/configs/farmsim/farmsim_occ_front3.py --launcher pytorch \
    --deterministic --work-dir "work_dirs/irwm_farmsim_front3_ep${EPOCHS}" \
    --load-from "${ROOT}/pretrained/r101_dcn_fcos3d_pretrain.pth" \
    --num-gpus "${NUM_GPUS}" --data-root "${DATA_ROOT}" --batch-size "${batch_size}" \
    --total-batch-size "${total_batch_size}" --image-width "${IMAGE_WIDTH}" --image-height "${IMAGE_HEIGHT}" --use-fp16 "${use_fp16}" \
    --workers-per-gpu "${WORKERS_PER_GPU}" \
    --history-frames 2 --predict-future-occ 1 --future-occ-steps 5 \
    --predict-future-traj 1 --future-traj-steps 6 --epochs "${EPOCHS}"
}

run_surroundocc() {
  local repo="${ROOT}/experiment/official/SurroundOcc"
  prepare_external_runtime
  cd "${repo}"
  # The upstream MMDet3D 0.17 source requires THC-era extensions and cannot
  # build with Torch 2.0.  ``surroundocc20`` uses its installed Torch2-ready
  # MMDet3D API; keeping the old source off PYTHONPATH is essential.
  # These variables are read while the FarmSim config is imported.  Pass them
  # on this exact launch command so a conda environment cannot hide a caller's
  # batch/image/precision choices.
  PYTHONPATH="${repo}:${ROOT}:${PYTHONPATH:-}" CUDA_HOME="${CUDA118_HOME}" CUDA_PATH="${CUDA118_HOME}" PATH="${CUDA118_HOME}/bin:${PATH}" LD_LIBRARY_PATH="${CUDA118_HOME}/lib64:${LD_LIBRARY_PATH:-}" CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}" \
    BATCH_SIZE="${FARMSIM_BATCH_SIZE}" TOTAL_BATCH_SIZE="${FARMSIM_TOTAL_BATCH_SIZE}" USE_FP16="${FARMSIM_USE_FP16}" IMAGE_WIDTH="${FARMSIM_IMAGE_WIDTH}" IMAGE_HEIGHT="${FARMSIM_IMAGE_HEIGHT}" WORKERS_PER_GPU="${WORKERS_PER_GPU}" \
    FARMSIM_BATCH_SIZE="${FARMSIM_BATCH_SIZE}" FARMSIM_TOTAL_BATCH_SIZE="${FARMSIM_TOTAL_BATCH_SIZE}" FARMSIM_MICRO_BATCH_SIZE="${FARMSIM_MICRO_BATCH_SIZE}" FARMSIM_WORLD_SIZE="${FARMSIM_WORLD_SIZE}" FARMSIM_GRAD_ACCUM_STEPS="${FARMSIM_GRAD_ACCUM_STEPS}" FARMSIM_IMAGE_WIDTH="${FARMSIM_IMAGE_WIDTH}" FARMSIM_IMAGE_HEIGHT="${FARMSIM_IMAGE_HEIGHT}" FARMSIM_USE_FP16="${FARMSIM_USE_FP16}" \
    conda run --no-capture-output -n "${SURROUNDOCC_ENV}" bash tools/dist_train.sh \
    projects/configs/surroundocc/surroundocc_farmsim_front3.py "${NUM_GPUS}" \
    "${ROOT}/work_dirs/surroundocc_farmsim_front3_ep${EPOCHS}" \
    --cfg-options total_epochs="${EPOCHS}" runner.max_epochs="${EPOCHS}"
}

run_cotr() {
  local repo="${ROOT}/experiment/official/COTR"
  prepare_external_runtime
  cd "${repo}"
  # COTR's config consumes the FARMSIM_* values at import time; keep the
  # user-facing names beside them to make every launch setting auditable.
  PYTHONPATH="${ROOT}:${repo}:${PYTHONPATH:-}" DRIVE_OCCWORLD_ROOT="${ROOT}" FARMSIM_DATA_ROOT="${DATA_ROOT}" CUDA_HOME="${CUDA118_HOME}" CUDA_PATH="${CUDA118_HOME}" PATH="${CUDA118_HOME}/bin:${PATH}" LD_LIBRARY_PATH="${CUDA118_HOME}/lib64:${LD_LIBRARY_PATH:-}" CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}" \
    BATCH_SIZE="${FARMSIM_BATCH_SIZE}" TOTAL_BATCH_SIZE="${FARMSIM_TOTAL_BATCH_SIZE}" USE_FP16="${FARMSIM_USE_FP16}" IMAGE_WIDTH="${FARMSIM_IMAGE_WIDTH}" IMAGE_HEIGHT="${FARMSIM_IMAGE_HEIGHT}" WORKERS_PER_GPU="${WORKERS_PER_GPU}" \
    FARMSIM_BATCH_SIZE="${FARMSIM_BATCH_SIZE}" FARMSIM_TOTAL_BATCH_SIZE="${FARMSIM_TOTAL_BATCH_SIZE}" FARMSIM_MICRO_BATCH_SIZE="${FARMSIM_MICRO_BATCH_SIZE}" FARMSIM_WORLD_SIZE="${FARMSIM_WORLD_SIZE}" FARMSIM_GRAD_ACCUM_STEPS="${FARMSIM_GRAD_ACCUM_STEPS}" FARMSIM_IMAGE_WIDTH="${FARMSIM_IMAGE_WIDTH}" FARMSIM_IMAGE_HEIGHT="${FARMSIM_IMAGE_HEIGHT}" FARMSIM_USE_FP16="${FARMSIM_USE_FP16}" \
    conda run --no-capture-output -n "${COTR_ENV}" python -m torch.distributed.launch \
    --use_env --nproc_per_node="${NUM_GPUS}" tools/train_occ.py \
    configs/cotr/cotr_farmsim_front3.py --launcher pytorch --seed 0 \
    --work-dir "${ROOT}/work_dirs/cotr_farmsim_front3_ep${EPOCHS}" \
    --cfg-options total_epochs="${EPOCHS}" runner.max_epochs="${EPOCHS}"
}

run_sparseocc() {
  local repo="${ROOT}/experiment/official/SparseOcc"
  # Do not use torchrun --standalone here.  On this host it can publish the
  # machine hostname as MASTER_ADDR; that hostname resolves to 127.0.1.1 and
  # has intermittently left non-zero ranks unable to reach the TCPStore.
  # A static loopback rendezvous is correct for this single-node launcher.
  local master_addr="${SPARSE_MASTER_ADDR:-127.0.0.1}"
  local master_port="${SPARSE_MASTER_PORT:-29541}"
  prepare_external_runtime
  cd "${repo}"
  # SparseOcc derives its one-step global batch from BATCH_SIZE x WORLD_SIZE
  # and uses FARMSIM_GRAD_ACCUM_STEPS for TOTAL_BATCH_SIZE.  Do not replace
  # FARMSIM_BATCH_SIZE with the effective batch here.
  PYTHONPATH="${repo}" CUDA_HOME="${CUDA118_HOME}" CUDA_PATH="${CUDA118_HOME}" PATH="${CUDA118_HOME}/bin:${PATH}" LD_LIBRARY_PATH="${CUDA118_HOME}/lib64:${LD_LIBRARY_PATH:-}" CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}" MASTER_ADDR="${master_addr}" MASTER_PORT="${master_port}" \
    BATCH_SIZE="${FARMSIM_BATCH_SIZE}" TOTAL_BATCH_SIZE="${FARMSIM_TOTAL_BATCH_SIZE}" USE_FP16="${FARMSIM_USE_FP16}" IMAGE_WIDTH="${FARMSIM_IMAGE_WIDTH}" IMAGE_HEIGHT="${FARMSIM_IMAGE_HEIGHT}" WORKERS_PER_GPU="${WORKERS_PER_GPU}" \
    FARMSIM_BATCH_SIZE="${FARMSIM_BATCH_SIZE}" FARMSIM_TOTAL_BATCH_SIZE="${FARMSIM_TOTAL_BATCH_SIZE}" FARMSIM_MICRO_BATCH_SIZE="${FARMSIM_MICRO_BATCH_SIZE}" FARMSIM_WORLD_SIZE="${FARMSIM_WORLD_SIZE}" FARMSIM_GRAD_ACCUM_STEPS="${FARMSIM_GRAD_ACCUM_STEPS}" FARMSIM_IMAGE_WIDTH="${FARMSIM_IMAGE_WIDTH}" FARMSIM_IMAGE_HEIGHT="${FARMSIM_IMAGE_HEIGHT}" FARMSIM_USE_FP16="${FARMSIM_USE_FP16}" \
    conda run --no-capture-output -n "${SPARSE_ENV}" python -m torch.distributed.run \
    --nnodes=1 --node_rank=0 --master_addr="${master_addr}" --master_port="${master_port}" --nproc_per_node="${NUM_GPUS}" train.py \
    --config configs/farmsim_front3_current.py \
    --run_name "farmsim_front3_ep${EPOCHS}" \
    --override total_epochs="${EPOCHS}"
}

case "${METHOD}" in
  # driveocc) run_native_driveocc ;;
  irwm) run_irwm ;;
  surroundocc) run_surroundocc ;;
  cotr) run_cotr ;;
  sparseocc) run_sparseocc ;;
  all)
    # run_native_driveocc

    # BATCH_SIZE=3
    # USE_FP16=0
    # run_cotr
    USE_FP16=1
    # BATCH_SIZE=6
    # # run_sparseocc
    BATCH_SIZE=2
    run_irwm
    # run_surroundocc
    ;;
  *)
    echo "Usage: $0 {driveocc|irwm|surroundocc|cotr|sparseocc|all}" >&2
    exit 2
    ;;
esac
