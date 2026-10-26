#!/usr/bin/env bash
# Existing checkpoint stays intact. Adapt only on original train/val captures.
set -euo pipefail
cd "$(dirname "$0")/.."
PYTHON_BIN="${PYTHON_BIN:-/home/HL/.conda/envs/dow2/bin/python}"
GPU_ID="${GPU_ID:-0}"
TASK_DIR="${1:-work_dirs/front3_s_base_agri_amoe96_cropsample0.1_min500_nohis_b24_ep15_20261012_214738}"
DATASET_ROOT="${2:-/data/HL/SimData-croprow}"
TASK_DIR="$(realpath "$TASK_DIR")"
ADAPT_DIR="$TASK_DIR/grid_calibration_finetune_20261020"
EVAL_DIR="$TASK_DIR/croprow_evaluation_20261020_grid_fixed"
export PYTHONPATH=".:${PYTHONPATH:-}"
export OMP_NUM_THREADS=2
export CUDA_VISIBLE_DEVICES="$GPU_ID"
GPU_MEMORY_MB="$(nvidia-smi --id="$GPU_ID" --query-gpu=memory.used --format=csv,noheader,nounits)"
if (( GPU_MEMORY_MB >= 500 )); then
  echo "GPU $GPU_ID is in use (${GPU_MEMORY_MB} MiB); choose a free GPU_ID." >&2
  exit 1
fi
if [[ ! -f "$ADAPT_DIR/selected_checkpoint.json" ]]; then
  "$PYTHON_BIN" row_det/prepare_grid_calibration_finetune.py \
    "$TASK_DIR/farmsim_occ_front3_s.py" "$TASK_DIR/epoch_15.pth" "$ADAPT_DIR"
  "$PYTHON_BIN" -u tools/train.py "$ADAPT_DIR/grid_calibration.py" --seed 0 --deterministic \
    > "$ADAPT_DIR/train.log" 2>&1
  "$PYTHON_BIN" row_det/select_grid_calibration_checkpoint.py "$ADAPT_DIR"
fi
SELECTED_CHECKPOINT="$("$PYTHON_BIN" -c 'import json,sys; print(json.load(open(sys.argv[1]))["selected"]["path"])' "$ADAPT_DIR/selected_checkpoint.json")"
"$PYTHON_BIN" row_det/prepare_croprow_evaluation.py "$DATASET_ROOT" "$EVAL_DIR"
"$PYTHON_BIN" -u tools/test.py "$ADAPT_DIR/grid_calibration.py" "$SELECTED_CHECKPOINT" \
  --out "$EVAL_DIR/occupancy_metrics.pkl" --save-predictions "$EVAL_DIR/prediction" \
  --save-prediction-count -1 --cfg-options \
  "data.test.ann_file=$EVAL_DIR/evaluation_manifest.json" "data.test.data_root=$DATASET_ROOT" \
  data.test.camera_geometry=grid_world data.test.samples_per_gpu=2 data.workers_per_gpu=2 \
  > "$EVAL_DIR/inference.log" 2>&1
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 "$PYTHON_BIN" -u row_det/evaluate_croprow.py "$EVAL_DIR" \
  --data-root "$DATASET_ROOT" --checkpoint "$SELECTED_CHECKPOINT" --workers 16 \
  --inference-config "$ADAPT_DIR/grid_calibration.py" --camera-geometry grid_world --inference-gpu "$GPU_ID" \
  > "$EVAL_DIR/row_evaluation.log" 2>&1
if [[ -f "$TASK_DIR/croprow_evaluation_20261020/row_metrics.json" && \
      -f "$TASK_DIR/croprow_evaluation_20261020_grid_world/row_metrics.json" ]]; then
  "$PYTHON_BIN" row_det/compare_croprow_repairs.py \
    "$TASK_DIR/croprow_evaluation_20261020" "$TASK_DIR/croprow_evaluation_20261020_grid_world" \
    "$EVAL_DIR" --adaptation-dir "$ADAPT_DIR" > "$EVAL_DIR/repair_comparison.log"
fi
