#!/usr/bin/env bash
# Batch-evaluate compatible Drive-OccWorld tasks and save prediction previews.
#
# Every direct child run under work_dirs is considered.  For each run, the
# numerically largest regular ``epoch_N.pth`` checkpoint is evaluated (EMA
# checkpoints are deliberately ignored).  Runs with a complete set of preview
# PNGs are skipped by default, so this script can safely be rerun.
#
# Each available GPU receives a serial queue of independent single-GPU test.py
# jobs. This avoids distributed rendezvous and allows unrelated experiments to
# evaluate concurrently. See the editable settings below or override them as
# environment variables, for example:
#   GPU_IDS=4,5 PREDICTION_COUNT=100 bash tools/evaluate_all_epoch8_with_previews.sh

set -uo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REPO_PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
PYTHON_BIN="${PYTHON_BIN:-/home/HL/.conda/envs/dow2/bin/python}"
WORK_DIRS_ROOT="${WORK_DIRS_ROOT:-${REPO_ROOT}/work_dirs}"
PREDICTION_COUNT="${PREDICTION_COUNT:-100}"
PREDICTION_DIR_NAME="${PREDICTION_DIR_NAME:-prediction}"
VISUALIZATION_DIR_NAME="${VISUALIZATION_DIR_NAME:-visualization}"
GT_CACHE_ROOT="${GT_CACHE_ROOT:-${WORK_DIRS_ROOT}/_shared_gt_previews}"
SAVE_SAMPLING="${SAVE_SAMPLING:-per-sequence}"
BATCH_SIZE="${BATCH_SIZE:-}"
MAX_USED_MEMORY_MIB="${MAX_USED_MEMORY_MIB:-1024}"
MAX_PARALLEL_GPU="${MAX_PARALLEL_GPU:-0}"
POLL_SECONDS="${POLL_SECONDS:-60}"
MAX_POINTS="${MAX_POINTS:-30000}"
PREVIEW_ELEV="${PREVIEW_ELEV:-26}"
PREVIEW_AZIM="${PREVIEW_AZIM:--135}"
RENDER_STYLE="${RENDER_STYLE:-auto}"
GPU_IDS="${GPU_IDS:-}"
DRY_RUN="${DRY_RUN:-0}"
SKIP_EXISTING_PREVIEWS="${SKIP_EXISTING_PREVIEWS:-1}"

if [[ ! -x "${PYTHON_BIN}" ]]; then
    echo "PYTHON_BIN is not executable: ${PYTHON_BIN}" >&2
    exit 2
fi
if [[ ! -d "${WORK_DIRS_ROOT}" ]]; then
    echo "WORK_DIRS_ROOT does not exist: ${WORK_DIRS_ROOT}" >&2
    exit 2
fi
if [[ ! "${PREDICTION_COUNT}" =~ ^[1-9][0-9]*$ ]]; then
    echo "PREDICTION_COUNT must be a positive integer" >&2
    exit 2
fi
if [[ "${SAVE_SAMPLING}" != "leading" && "${SAVE_SAMPLING}" != "per-sequence" ]]; then
    echo "SAVE_SAMPLING must be leading or per-sequence" >&2
    exit 2
fi

find_config() {
    local run_dir="$1"
    local configs=()
    while IFS= read -r -d '' config; do
        configs+=("${config}")
    done < <(find "${run_dir}" -maxdepth 1 -type f -name '*.py' -print0 | sort -z)
    if (( ${#configs[@]} != 1 )); then
        echo "Expected exactly one dumped config in ${run_dir}, found ${#configs[@]}" >&2
        return 1
    fi
    printf '%s\n' "${configs[0]}"
}

dataset_key() {
    local config="$1"
    # This script is commonly launched through non-interactive shells where
    # user-local tools such as ripgrep are not guaranteed to be on PATH.
    # POSIX grep is sufficient here and accepts both compact and spaced MMCV
    # config syntax (for example, ``type='...'`` and ``type = '...'``).
    if grep -Eq "type[[:space:]]*=[[:space:]]*['\"]FarmSimWorldDataset['\"]" "${config}"; then
        printf 'farmsim\n'
    elif grep -Eq "type[[:space:]]*=[[:space:]]*['\"]ORAD3DWorldDataset['\"]" "${config}"; then
        printf 'orad3d\n'
    elif grep -Eq "type[[:space:]]*=[[:space:]]*['\"]FarmSimSurroundOccDataset['\"]" "${config}"; then
        printf 'external-surroundocc\n'
    else
        printf 'unknown\n'
    fi
}

evaluation_ann_file() {
    case "$1" in
        farmsim) printf 'data/farmsim/splits/val.json\n' ;;
        simdata-occ0p1) printf 'data/simdata_occ0p1/splits/val.json\n' ;;
        orad3d) printf 'data/orad3d/splits/test.json\n' ;;
        *) return 1 ;;
    esac
}

# Return the most recent regular training checkpoint in a run directory.  Do
# not select epoch_N_ema.pth: it is a different checkpoint format and test.py
# should receive the ordinary epoch_N.pth produced by the training run.
latest_checkpoint() {
    local run_dir="$1" checkpoint basename epoch=-1 latest=''
    while IFS= read -r -d '' checkpoint; do
        basename="$(basename "${checkpoint}")"
        if [[ "${basename}" =~ ^epoch_([0-9]+)\.pth$ ]] \
            && (( 10#${BASH_REMATCH[1]} > epoch )); then
            epoch=$((10#${BASH_REMATCH[1]}))
            latest="${checkpoint}"
        fi
    done < <(find "${run_dir}" -maxdepth 1 -type f -name 'epoch_*.pth' -print0)
    [[ -n "${latest}" ]] && printf '%s\n' "${latest}"
}

preview_count() {
    local run_dir="$1"
    find "${run_dir}/${VISUALIZATION_DIR_NAME}" -maxdepth 1 -type f \
        -name '*_prediction.png' 2>/dev/null | wc -l
}

# A run directory is queued at most once, using its highest epoch_N.pth.
# Existing preview images are the completion marker because they represent the
# user-visible result requested by this evaluator; prediction NPZ files alone
# are intentionally not sufficient to suppress re-rendering.
declare -a TASK_CHECKPOINTS=()
SKIPPED_EXISTING_PREVIEWS=0
while IFS= read -r -d '' run_dir; do
    checkpoint="$(latest_checkpoint "${run_dir}")"
    [[ -z "${checkpoint}" ]] && continue
    existing_previews="$(preview_count "${run_dir}")"
    if [[ "${SKIP_EXISTING_PREVIEWS}" == "1" ]] \
        && (( existing_previews >= PREDICTION_COUNT )); then
        echo "[$(basename "${run_dir}")] skipped: ${existing_previews} existing preview image(s)"
        SKIPPED_EXISTING_PREVIEWS=$((SKIPPED_EXISTING_PREVIEWS + 1))
        continue
    fi
    TASK_CHECKPOINTS+=("${checkpoint}")
done < <(find "${WORK_DIRS_ROOT}" -mindepth 1 -maxdepth 1 -type d -print0 | sort -z)

if (( ${#TASK_CHECKPOINTS[@]} == 0 )); then
    echo "No unevaluated epoch_N.pth checkpoints found directly under ${WORK_DIRS_ROOT}." >&2
    exit 0
fi

run_task() {
    local gpu="$1" checkpoint="$2"
    local run_dir checkpoint_stem
    run_dir="$(dirname "${checkpoint}")"
    checkpoint_stem="$(basename "${checkpoint%.pth}")"
    local config dataset ann_file prediction_dir visualization_dir gt_dir log_path metrics_path
    if [[ ! -f "${checkpoint}" ]]; then
        echo "[$(basename "${run_dir}")] skipped: checkpoint not found: ${checkpoint}" >&2
        return 1
    fi
    config="$(find_config "${run_dir}")" || return 1
    dataset="$(dataset_key "${config}")"
    if [[ "$(basename "${run_dir}")" == simdata_* ]]; then
        dataset="simdata-occ0p1"
    fi
    if [[ "${dataset}" == "external-surroundocc" ]]; then
        echo "[$(basename "${run_dir}")] skipped: external SurroundOcc needs its own Pkl-based evaluator and does not emit Drive-OccWorld NPZ artifacts"
        return 0
    fi
    if [[ "${dataset}" == "unknown" ]]; then
        echo "[$(basename "${run_dir}")] skipped: unsupported dataset config ${config}" >&2
        return 0
    fi
    ann_file="$(evaluation_ann_file "${dataset}")" || return 1
    prediction_dir="${run_dir}/${PREDICTION_DIR_NAME}"
    visualization_dir="${run_dir}/${VISUALIZATION_DIR_NAME}"
    gt_dir="${GT_CACHE_ROOT}/${dataset}"
    log_path="${run_dir}/evaluation_${checkpoint_stem}.log"
    metrics_path="${run_dir}/evaluation_${checkpoint_stem}_metrics.pkl"

    echo "[$(date '+%F %T')] GPU ${gpu}: $(basename "${run_dir}") (${dataset}, ${checkpoint_stem})"
    if [[ "${DRY_RUN}" == "1" ]]; then
        printf '  config=%s\n  checkpoint=%s\n  ann_file=%s\n  prediction=%s\n  visualization=%s\n' \
            "${config}" "${checkpoint}" "${ann_file}" "${prediction_dir}" "${visualization_dir}"
        return 0
    fi

    mkdir -p "${prediction_dir}" "${visualization_dir}" "${gt_dir}"
    local existing_count
    existing_count="$(find "${prediction_dir}" -maxdepth 1 -type f -name '*.npz' | wc -l)"
    if (( existing_count == PREDICTION_COUNT )); then
        echo "  found ${existing_count} existing prediction artifacts; skipping inference"
    else
        if (( existing_count > 0 )); then
            echo "  removing ${existing_count} incomplete/stale prediction artifact(s) before re-evaluation"
            find "${prediction_dir}" -maxdepth 1 -type f -name '*.npz' -delete
        fi
        find "${prediction_dir}" -maxdepth 1 -type f -name '.epoch8_batch_eval_manifest' -delete

        local command=("${PYTHON_BIN}" tools/test.py "${config}" "${checkpoint}"
            --save-predictions "${prediction_dir}"
            --save-prediction-count "${PREDICTION_COUNT}"
            --save-prediction-sampling "${SAVE_SAMPLING}"
            --cfg-options "data.test.ann_file=${ann_file}"
            --out "${metrics_path}")
        if [[ -n "${BATCH_SIZE}" ]]; then
            command+=(--batch-size "${BATCH_SIZE}")
        fi
        echo "  running inference; log: ${log_path}"
        if ! (cd "${REPO_ROOT}" && PYTHONPATH="${REPO_PYTHONPATH}" CUDA_VISIBLE_DEVICES="${gpu}" "${command[@]}") \
                >"${log_path}" 2>&1; then
            echo "  evaluation failed; inspect ${log_path}" >&2
            return 1
        fi
        local saved_count
        saved_count="$(find "${prediction_dir}" -maxdepth 1 -type f -name '*.npz' | wc -l)"
        if (( saved_count != PREDICTION_COUNT )); then
            echo "  expected ${PREDICTION_COUNT} artifacts, found ${saved_count}; see ${log_path}" >&2
            return 1
        fi
    fi

    echo "  rendering ${PREDICTION_COUNT} prediction-only previews; GT cache: ${gt_dir}"
    local render_command=("${PYTHON_BIN}" "${REPO_ROOT}/tools/render_occ_prediction_previews.py"
        "${prediction_dir}" "${visualization_dir}" --shared-gt-dir "${gt_dir}"
        --count "${PREDICTION_COUNT}" --max-points "${MAX_POINTS}"
        --elev "${PREVIEW_ELEV}" --azim "${PREVIEW_AZIM}"
        --style "${RENDER_STYLE}")
    render_command+=(--overwrite)
    if ! "${render_command[@]}"; then
        echo "  preview rendering failed" >&2
        return 1
    fi
    echo "  complete: ${prediction_dir}, ${visualization_dir}"
}

available_gpus() {
    local allowed=",${GPU_IDS},"
    while IFS=',' read -r index used total utilization; do
        index="${index//[[:space:]]/}"
        used="${used//[[:space:]]/}"
        if [[ -n "${GPU_IDS}" && "${allowed}" != *",${index},"* ]]; then
            continue
        fi
        if [[ "${used}" =~ ^[0-9]+$ ]] && (( used <= MAX_USED_MEMORY_MIB )); then
            printf '%s\n' "${index}"
        fi
    done < <(nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu \
        --format=csv,noheader,nounits 2>/dev/null)
}

while :; do
    mapfile -t AVAILABLE_GPUS < <(available_gpus)
    if (( ${#AVAILABLE_GPUS[@]} > 0 )); then
        break
    fi
    if [[ -n "${GPU_IDS}" ]]; then
        echo "No requested GPU in GPU_IDS=${GPU_IDS} is below ${MAX_USED_MEMORY_MIB} MiB; retrying in ${POLL_SECONDS}s..."
    else
        echo "No GPU is below ${MAX_USED_MEMORY_MIB} MiB; retrying in ${POLL_SECONDS}s..."
    fi
    sleep "${POLL_SECONDS}"
done
if (( MAX_PARALLEL_GPU > 0 && ${#AVAILABLE_GPUS[@]} > MAX_PARALLEL_GPU )); then
    AVAILABLE_GPUS=("${AVAILABLE_GPUS[@]:0:MAX_PARALLEL_GPU}")
fi

echo "Found ${#TASK_CHECKPOINTS[@]} unevaluated task(s), skipped ${SKIPPED_EXISTING_PREVIEWS} with existing previews, and ${#AVAILABLE_GPUS[@]} available GPU(s): ${AVAILABLE_GPUS[*]}"
echo "Each task saves ${PREDICTION_COUNT} ${SAVE_SAMPLING} samples. Shared GT cache: ${GT_CACHE_ROOT}"

declare -a WORKER_TASKS
for ((index = 0; index < ${#AVAILABLE_GPUS[@]}; index++)); do
    WORKER_TASKS[index]=''
done
for ((index = 0; index < ${#TASK_CHECKPOINTS[@]}; index++)); do
    worker=$((index % ${#AVAILABLE_GPUS[@]}))
    WORKER_TASKS[worker]+="${TASK_CHECKPOINTS[index]}"$'\n'
done

worker() {
    local gpu="$1" tasks="$2" checkpoint failures=0
    while IFS= read -r checkpoint; do
        [[ -z "${checkpoint}" ]] && continue
        run_task "${gpu}" "${checkpoint}" || failures=$((failures + 1))
    done <<< "${tasks}"
    return "${failures}"
}

declare -a WORKER_PIDS=()
for ((index = 0; index < ${#AVAILABLE_GPUS[@]}; index++)); do
    worker "${AVAILABLE_GPUS[index]}" "${WORKER_TASKS[index]}" &
    WORKER_PIDS+=("$!")
done

FAILURES=0
for pid in "${WORKER_PIDS[@]}"; do
    wait "${pid}" || FAILURES=$((FAILURES + 1))
done
if (( FAILURES > 0 )); then
    echo "Finished with ${FAILURES} worker(s) containing failed task(s)." >&2
    exit 1
fi
echo "All selected evaluation and preview tasks completed."
