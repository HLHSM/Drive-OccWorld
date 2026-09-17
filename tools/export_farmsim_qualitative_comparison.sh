#!/usr/bin/env bash
# Export one-to-one qualitative FarmSim predictions for AgriOcc, IR-WM, and
# SurroundOcc.  The AgriOcc reference set and its GT previews are prepared by
# tools/prepare_qualitative_comparison.py and render_occ_prediction_previews.py.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
GPU_ID="${GPU_ID:-0}"
DOW_PYTHON="${DOW_PYTHON:-/home/HL/.conda/envs/dow2/bin/python}"
SURROUNDOCC_PYTHON="${SURROUNDOCC_PYTHON:-/home/HL/.conda/envs/surroundocc20/bin/python}"
COMPARISON_ROOT="${REPO_ROOT}/work_dirs/qualitative_comparison"
AGRIOCC_PREDICTION_DIR="${REPO_ROOT}/work_dirs/front3_gvad_agri_amoe_nearfar_r0.6_s2_ep8_20260911_105909/prediction"
IRWM_RUN_DIR="${REPO_ROOT}/work_dirs/irwm_farmsim_front3_ep8"
SURROUNDOCC_RUN_DIR="${REPO_ROOT}/work_dirs/surroundocc_farmsim_front3_ep8"
OFFICIAL_SURROUNDOCC_ROOT="${REPO_ROOT}/experiment/official/SurroundOcc"
MANIFEST="${COMPARISON_ROOT}/manifest.json"
INDEX_FILE="${COMPARISON_ROOT}/selected_indices.txt"
IRWM_INDEX_FILE="${COMPARISON_ROOT}/irwm_selected_indices.txt"
SURROUNDOCC_INFO="${COMPARISON_ROOT}/surroundocc_val_selected.pkl"
GT_VISUALIZATION_DIR="${COMPARISON_ROOT}/gt_visualization"
IRWM_PREDICTION_DIR="${COMPARISON_ROOT}/irwm_prediction_matched"
IRWM_VISUALIZATION_DIR="${COMPARISON_ROOT}/irwm_visualization_matched"
SURROUNDOCC_RAW_DIR="${COMPARISON_ROOT}/surroundocc_raw"
SURROUNDOCC_PREDICTION_DIR="${COMPARISON_ROOT}/surroundocc_prediction_matched"
SURROUNDOCC_VISUALIZATION_DIR="${COMPARISON_ROOT}/surroundocc_visualization_matched"

require_empty_dir() {
    local dir="$1"
    if [[ -e "$dir" ]] && find "$dir" -mindepth 1 -print -quit | grep -q .; then
        echo "Refusing to mix an earlier export with this run: $dir" >&2
        echo "Choose a new COMPARISON_ROOT or inspect the directory before removing it." >&2
        exit 2
    fi
    mkdir -p "$dir"
}

for required in "$MANIFEST" "$INDEX_FILE" "$IRWM_INDEX_FILE" "$SURROUNDOCC_INFO"; do
    [[ -f "$required" ]] || { echo "Missing prerequisite: $required" >&2; exit 2; }
done
command -v nvidia-smi >/dev/null || { echo 'nvidia-smi is unavailable.' >&2; exit 2; }
nvidia-smi -L >/dev/null || { echo 'No CUDA GPU is visible to this shell.' >&2; exit 2; }

require_empty_dir "$IRWM_PREDICTION_DIR"
require_empty_dir "$IRWM_VISUALIZATION_DIR"
require_empty_dir "$SURROUNDOCC_RAW_DIR"
require_empty_dir "$SURROUNDOCC_PREDICTION_DIR"
require_empty_dir "$SURROUNDOCC_VISUALIZATION_DIR"

echo '[1/4] Exporting IR-WM on the AgriOcc sample indices.'
(
    cd "$REPO_ROOT"
    PYTHONPATH="$REPO_ROOT${PYTHONPATH:+:${PYTHONPATH}}" CUDA_VISIBLE_DEVICES="$GPU_ID" \
        "$DOW_PYTHON" tools/test.py "$IRWM_RUN_DIR/farmsim_occ_front3.py" "$IRWM_RUN_DIR/epoch_8.pth" \
        --save-predictions "$IRWM_PREDICTION_DIR" \
        --save-prediction-indices "$IRWM_INDEX_FILE" \
        --prediction-subset-only \
        --prediction-reference-indices "$INDEX_FILE" \
        --out "$COMPARISON_ROOT/irwm_matched_metrics.pkl"
)

echo '[2/4] Rendering IR-WM prediction candidates.'
MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/mplconfig}" "$DOW_PYTHON" \
    "$REPO_ROOT/tools/render_occ_prediction_previews.py" \
    "$IRWM_PREDICTION_DIR" "$IRWM_VISUALIZATION_DIR" \
    --shared-gt-dir "$GT_VISUALIZATION_DIR" --count 100 --max-points 30000 \
    --elev 26 --azim -135 --skip-gt

echo '[3/4] Exporting official SurroundOcc on the same 100 FarmSim frames.'
(
    cd "$OFFICIAL_SURROUNDOCC_ROOT"
    PYTHONPATH="$OFFICIAL_SURROUNDOCC_ROOT${PYTHONPATH:+:${PYTHONPATH}}" \
        CUDA_VISIBLE_DEVICES="$GPU_ID" SURROUNDOCC_VIS_DIR="$SURROUNDOCC_RAW_DIR" \
        "$SURROUNDOCC_PYTHON" -m torch.distributed.run --nproc_per_node=1 \
        tools/test.py "$SURROUNDOCC_RUN_DIR/surroundocc_farmsim_front3.py" "$SURROUNDOCC_RUN_DIR/epoch_8.pth" \
        --launcher pytorch --is_vis --show-dir "$SURROUNDOCC_RAW_DIR" \
        --cfg-options "model.is_vis=True" "data.test.ann_file=$SURROUNDOCC_INFO" \
        "data.workers_per_gpu=0"
)

"$DOW_PYTHON" "$REPO_ROOT/tools/convert_surroundocc_visuals.py" \
    "$MANIFEST" "$AGRIOCC_PREDICTION_DIR" "$SURROUNDOCC_RAW_DIR" "$SURROUNDOCC_PREDICTION_DIR"

echo '[4/4] Rendering SurroundOcc prediction candidates.'
MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/mplconfig}" "$DOW_PYTHON" \
    "$REPO_ROOT/tools/render_occ_prediction_previews.py" \
    "$SURROUNDOCC_PREDICTION_DIR" "$SURROUNDOCC_VISUALIZATION_DIR" \
    --shared-gt-dir "$GT_VISUALIZATION_DIR" --count 100 --max-points 30000 \
    --elev 26 --azim -135 --skip-gt

echo 'Completed matched exports:'
printf '  AgriOcc: %s\n  IR-WM: %s\n  SurroundOcc: %s\n  GT: %s\n' \
    "$COMPARISON_ROOT/agriocc_visualization" "$IRWM_VISUALIZATION_DIR" \
    "$SURROUNDOCC_VISUALIZATION_DIR" "$GT_VISUALIZATION_DIR"
