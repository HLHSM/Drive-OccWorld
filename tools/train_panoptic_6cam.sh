#!/usr/bin/env bash
# Main model: six cameras + GVAD + Agri-AMoE + thing query masks, native 11 classes.
# 0.2m, x=[-20,20], y=[-10,10], z=[-2,3]; train stride defaults to 2.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
export PYTHONPATH="${PWD}:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1
PYTHON_BIN="${PYTHON_BIN:-/home/HL/.conda/envs/dow2/bin/python}"
NUM_GPUS="${NUM_GPUS:-4}"
DATA_ROOT="${DATA_ROOT:-/data/HL/SimData-panoptic/SimData-panoptic}"
SPLIT_DIR="${SPLIT_DIR:-/data/HL/FlashOCC/data/simdata/splits}"
WORK_DIR="${WORK_DIR:-work_dirs/panoptic_6cam_tsa_amoe_query_$(date +%Y%m%d_%H%M%S)}"
BATCH_SIZE="${BATCH_SIZE:-4}"
TOTAL_BATCH_SIZE="${TOTAL_BATCH_SIZE:-16}"
EPOCHS="${EPOCHS:-15}"
NUM_QUERIES="${NUM_QUERIES:-256}"
SAMPLE_STRIDE="${SAMPLE_STRIDE:-2}"

# Read FlashOCC split files directly; check the frames selected for training.
"${PYTHON_BIN}" tools/check_panoptic_training_data.py \
  --manifest "${SPLIT_DIR}/train.json" --data-root "${DATA_ROOT}" --num-queries "${NUM_QUERIES}" \
  --sample-stride "${SAMPLE_STRIDE}" \
  --output "work_dirs/panoptic_6cam_flashocc_stride${SAMPLE_STRIDE}_training_data.json"
mkdir -p "${WORK_DIR}"
"${PYTHON_BIN}" -m torch.distributed.run --standalone --nproc_per_node="${NUM_GPUS}" \
  --log-dir "${WORK_DIR}/torchrun" --tee 3 \
  tools/train.py projects/configs/farmsim/farmsim_panoptic_6cam.py \
  --launcher pytorch --deterministic --seed 0 --work-dir "${WORK_DIR}" \
  --load-from pretrained/r101_dcn_fcos3d_pretrain.pth \
  --num-gpus "${NUM_GPUS}" --data-root "${DATA_ROOT}" \
  --train-ann-file "${SPLIT_DIR}/train.json" \
  --batch-size "${BATCH_SIZE}" --total-batch-size "${TOTAL_BATCH_SIZE}" \
  --workers-per-gpu 4 --image-width 512 --image-height 288 --camera-geometry grid_world \
  --use-fp16 1 --fp16-initial-scale 128 --epochs "${EPOCHS}" --history-frames 0 \
  --predict-future-occ 0 --predict-future-traj 0 --use-r50-image-encoder 0 \
  --use-gvad-attention 0 --gvad-use-visibility 1 --gvad-use-local-deformable 1 \
  --gvad-num-heads 8 --gvad-anchor-grid-height 4 --gvad-anchor-grid-width 8 \
  --cfg-options log_config.interval=20 evaluation.interval=1 \
  "model.future_pred_head.instance_decoder.num_queries=${NUM_QUERIES}" \
  "data.train.sample_stride=${SAMPLE_STRIDE}" \
  "data.val.ann_file=${SPLIT_DIR}/val.json" "data.test.ann_file=${SPLIT_DIR}/test.json" \
  2>&1 | tee "${WORK_DIR}/launcher.log"
