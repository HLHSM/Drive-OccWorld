#!/usr/bin/env bash
# Batch-evaluate compatible Drive-OccWorld tasks with epoch_8.pth.
#
# Each available GPU receives a serial queue of independent single-GPU test.py
# jobs. This avoids distributed rendezvous and allows unrelated experiments to
# evaluate concurrently. See the editable settings below or override them as
# environment variables, for example:
#   GPU_IDS=4,5 PREDICTION_COUNT=100 bash tools/evaluate_all_epoch8_with_previews.sh

set -uo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/home/HL/.conda/envs/dow2/bin/python}"
WORK_DIRS_ROOT="${WORK_DIRS_ROOT:-${REPO_ROOT}/work_dirs}"
CHECKPOINT_NAME="${CHECKPOINT_NAME:-epoch_8.pth}"
PREDICTION_COUNT="${PREDICTION_COUNT:-100}"
PREDICTION_DIR_NAME="${PREDICTION_DIR_NAME:-prediction}"
VISUALIZATION_DIR_NAME="${VISUALIZATION_DIR_NAME:-visualization}"
GT_CACHE_ROOT="${GT_CACHE_ROOT:-${WORK_DIRS_ROOT}/_shared_gt_previews}"
SAVE_SAMPLING="${SAVE_SAMPLING:-per-sequence}"
BATCH_SIZE="${BATCH_SIZE:-}"
MAX_USED_MEMORY_MIB="${MAX_USED_MEMORY_MIB:-1024}"
MAX_PARALLEL_GPU="${MAX_PARALLEL_GPU:-0}"
POLL_SECONDS="${POLL_SECONDS:-60}"
MAX_POINTS="${MAX_POINTS:-16000}"
PREVIEW_ELEV="${PREVIEW_ELEV:-22}"
PREVIEW_AZIM="${PREVIEW_AZIM:--58}"
GPU_IDS="${GPU_IDS:-}"
FORCE="${FORCE:-0}"
DRY_RUN="${DRY_RUN:-0}"

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

declare -a TASK_DIRS=()
while IFS= read -r -d '' checkpoint; do
    TASK_DIRS+=("$(dirname "${checkpoint}")")
done < <(find "${WORK_DIRS_ROOT}" -mindepth 2 -maxdepth 2 -type f \
    -name "${CHECKPOINT_NAME}" -print0 | sort -z)

if (( ${#TASK_DIRS[@]} == 0 )); then
    echo "No ${CHECKPOINT_NAME} files found directly under ${WORK_DIRS_ROOT}" >&2
    exit 0
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
    if rg -q "type='FarmSimWorldDataset'|type=\"FarmSimWorldDataset\"" "${config}"; then
        printf 'farmsim\n'
    elif rg -q "type='ORAD3DWorldDataset'|type=\"ORAD3DWorldDataset\"" "${config}"; then
        printf 'orad3d\n'
    elif rg -q "type='FarmSimSurroundOccDataset'|type=\"FarmSimSurroundOccDataset\"" "${config}"; then
        printf 'external-surroundocc\n'
    else
        printf 'unknown\n'
    fi
}

prediction_manifest_matches() {
    local run_dir="$1" config="$2" prediction_dir="$3" dataset="$4" ann_file="$5"
    local manifest="${prediction_dir}/.epoch8_batch_eval_manifest"
    [[ -f "${manifest}" ]] || return 1
    [[ $(find "${prediction_dir}" -maxdepth 1 -type f -name '*.npz' | wc -l) -eq "${PREDICTION_COUNT}" ]] || return 1
    grep -Fqx "checkpoint=${run_dir}/${CHECKPOINT_NAME}" "${manifest}" && \
        grep -Fqx "config=${config}" "${manifest}" && \
        grep -Fqx "sampling=${SAVE_SAMPLING}" "${manifest}" && \
        grep -Fqx "count=${PREDICTION_COUNT}" "${manifest}" && \
        grep -Fqx "dataset=${dataset}" "${manifest}" && \
        grep -Fqx "ann_file=${ann_file}" "${manifest}"
}

write_manifest() {
    local run_dir="$1" config="$2" prediction_dir="$3" dataset="$4" ann_file="$5"
    local manifest="${prediction_dir}/.epoch8_batch_eval_manifest"
    {
        printf 'checkpoint=%s/%s\n' "${run_dir}" "${CHECKPOINT_NAME}"
        printf 'config=%s\n' "${config}"
        printf 'sampling=%s\n' "${SAVE_SAMPLING}"
        printf 'count=%s\n' "${PREDICTION_COUNT}"
        printf 'dataset=%s\n' "${dataset}"
        printf 'ann_file=%s\n' "${ann_file}"
    } > "${manifest}"
}

evaluation_ann_file() {
    case "$1" in
        farmsim) printf 'data/farmsim/splits/val.json\n' ;;
        orad3d) printf 'data/orad3d/splits/test.json\n' ;;
        *) return 1 ;;
    esac
}

run_task() {
    local gpu="$1" run_dir="$2"
    local checkpoint="${run_dir}/${CHECKPOINT_NAME}"
    local config dataset ann_file prediction_dir visualization_dir gt_dir log_path metrics_path
    config="$(find_config "${run_dir}")" || return 1
    dataset="$(dataset_key "${config}")"
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
    log_path="${run_dir}/evaluation_${CHECKPOINT_NAME%.pth}.log"
    metrics_path="${run_dir}/evaluation_${CHECKPOINT_NAME%.pth}_metrics.pkl"

    echo "[$(date '+%F %T')] GPU ${gpu}: $(basename "${run_dir}") (${dataset})"
    if [[ "${DRY_RUN}" == "1" ]]; then
        printf '  config=%s\n  checkpoint=%s\n  ann_file=%s\n  prediction=%s\n  visualization=%s\n' \
            "${config}" "${checkpoint}" "${ann_file}" "${prediction_dir}" "${visualization_dir}"
        return 0
    fi

    mkdir -p "${prediction_dir}" "${visualization_dir}" "${gt_dir}"
    if prediction_manifest_matches "${run_dir}" "${config}" "${prediction_dir}" "${dataset}" "${ann_file}"; then
        if [[ "${FORCE}" != "1" ]]; then
            echo "  predictions already complete; skipping inference"
        else
            echo "  FORCE=1; regenerating the verified prediction set"
            local command=("${PYTHON_BIN}" tools/test.py "${config}" "${checkpoint}"
                --save-predictions "${prediction_dir}"
                --save-prediction-count "${PREDICTION_COUNT}"
                --save-prediction-sampling "${SAVE_SAMPLING}"
                --cfg-options "data.test.ann_file=${ann_file}"
                --out "${metrics_path}")
            if [[ -n "${BATCH_SIZE}" ]]; then
                command+=(--batch-size "${BATCH_SIZE}")
            fi
            if ! (cd "${REPO_ROOT}" && CUDA_VISIBLE_DEVICES="${gpu}" "${command[@]}") \
                    >"${log_path}" 2>&1; then
                echo "  evaluation failed; inspect ${log_path}" >&2
                return 1
            fi
        fi
    else
        local existing_count
        existing_count="$(find "${prediction_dir}" -maxdepth 1 -type f -name '*.npz' | wc -l)"
        if (( existing_count > 0 )); then
            echo "  ${prediction_dir} contains ${existing_count} unverified NPZ files. Refusing to mix artifacts; move it aside or use a different PREDICTION_DIR_NAME." >&2
            return 1
        fi
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
        if ! (cd "${REPO_ROOT}" && CUDA_VISIBLE_DEVICES="${gpu}" "${command[@]}") \
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
        write_manifest "${run_dir}" "${config}" "${prediction_dir}" "${dataset}" "${ann_file}"
    fi

    echo "  rendering ${PREDICTION_COUNT} prediction-only previews; GT cache: ${gt_dir}"
    local render_command=("${PYTHON_BIN}" "${REPO_ROOT}/tools/render_occ_prediction_previews.py"
        "${prediction_dir}" "${visualization_dir}" --shared-gt-dir "${gt_dir}"
        --count "${PREDICTION_COUNT}" --max-points "${MAX_POINTS}"
        --elev "${PREVIEW_ELEV}" --azim "${PREVIEW_AZIM}")
    if [[ "${FORCE}" == "1" ]]; then
        render_command+=(--overwrite)
    fi
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

echo "Found ${#TASK_DIRS[@]} tasks and ${#AVAILABLE_GPUS[@]} available GPU(s): ${AVAILABLE_GPUS[*]}"
echo "Each task saves ${PREDICTION_COUNT} ${SAVE_SAMPLING} samples. Shared GT cache: ${GT_CACHE_ROOT}"

declare -a WORKER_TASKS
for ((index = 0; index < ${#AVAILABLE_GPUS[@]}; index++)); do
    WORKER_TASKS[index]=''
done
for ((index = 0; index < ${#TASK_DIRS[@]}; index++)); do
    worker=$((index % ${#AVAILABLE_GPUS[@]}))
    WORKER_TASKS[worker]+="${TASK_DIRS[index]}"$'\n'
done

worker() {
    local gpu="$1" tasks="$2" run_dir failures=0
    while IFS= read -r run_dir; do
        [[ -z "${run_dir}" ]] && continue
        run_task "${gpu}" "${run_dir}" || failures=$((failures + 1))
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
echo "All epoch-8 evaluation and preview tasks completed."
