#!/bin/bash
set -Eeuo pipefail

PYTHON_BIN="$(which python)"
TORCHRUN_BIN="$(which torchrun)"
ACCELERATE_BIN="$(which accelerate)"

if [[ ! -x "${PYTHON_BIN}" ]]; then
    echo "[FATAL] python not found in current environment"
    exit 1
fi
if [[ ! -x "${TORCHRUN_BIN}" ]]; then
    echo "[FATAL] torchrun not found in current environment"
    exit 1
fi
if [[ ! -x "${ACCELERATE_BIN}" ]]; then
    echo "[FATAL] accelerate not found in current environment"
    exit 1
fi

RUN_ID=$(date +"%Y%m%d_%H%M%S")
ROOT_DIR="./experiments/runs/${RUN_ID}"
LOG_DIR="${ROOT_DIR}/logs"
SUMMARY_DIR="${ROOT_DIR}/summary"
CSV_DIR="${ROOT_DIR}/csv"

mkdir -p "${ROOT_DIR}" "${LOG_DIR}" "${SUMMARY_DIR}" "${CSV_DIR}"

CATEGORY="Industrial_and_Scientific"
TRAIN_FILE="./data/Amazon/train/${CATEGORY}_5_2016-10-2018-11.csv"
EVAL_FILE="./data/Amazon/valid/${CATEGORY}_5_2016-10-2018-11.csv"
TEST_FILE="./data/Amazon/test/${CATEGORY}_5_2016-10-2018-11.csv"
INFO_FILE="./data/Amazon/info/${CATEGORY}_5_2016-10-2018-11.txt"
SID_INDEX_PATH="./data/Amazon/index/${CATEGORY}.index.json"
ITEM_META_PATH="./data/Amazon/index/${CATEGORY}.item.json"

# ==========================================
# 🌟 核心修改区：8卡全开，霸占资源，完美整除
# ==========================================
export CUDA_VISIBLE_DEVICES="2,3,4,5,6,7,8,9"
NUM_GPUS=8
MASTER_PORT=29600

# SFT 阶段：16(微批次) * 8(卡) * 4(累加步数) = 512
SFT_GLOBAL_BATCH=512
SFT_MICRO_BATCH=16

# RL 阶段参数
RL_TRAIN_BATCH=32
RL_EVAL_BATCH=64
RL_NUM_GENERATIONS=16
RL_BEAM_SEARCH=True
RL_NUM_EPOCHS=2
RL_GRAD_ACC=2
# ==========================================

export NCCL_IB_DISABLE=1
export NCCL_P2P_DISABLE=1
export PYDANTIC_SKIP_VALIDATION=1
export OMP_NUM_THREADS=1
export TOKENIZERS_PARALLELISM=false
export WANDB_MODE=offline
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

RUN_ALL_LOG="${LOG_DIR}/run_all.log"

log() {
    echo "[$(date '+%F %T')] $*" | tee -a "${RUN_ALL_LOG}"
}

die() {
    echo "" | tee -a "${RUN_ALL_LOG}"
    echo "========================================" | tee -a "${RUN_ALL_LOG}"
    echo "ERROR DETECTED" | tee -a "${RUN_ALL_LOG}"
    echo "Line      : ${1:-unknown}" | tee -a "${RUN_ALL_LOG}"
    echo "Command   : ${2:-unknown}" | tee -a "${RUN_ALL_LOG}"
    echo "Exit Code : ${3:-unknown}" | tee -a "${RUN_ALL_LOG}"
    echo "Log File  : ${RUN_ALL_LOG}" | tee -a "${RUN_ALL_LOG}"
    echo "========================================" | tee -a "${RUN_ALL_LOG}"
    exit 1
}

trap 'die "${LINENO}" "${BASH_COMMAND}" "$?"' ERR

run_and_log() {
    local log_file="$1"
    shift

    mkdir -p "$(dirname "${log_file}")"

    echo "" | tee -a "${RUN_ALL_LOG}"
    echo "----------------------------------------" | tee -a "${RUN_ALL_LOG}"
    echo "[RUN] $*" | tee -a "${RUN_ALL_LOG}"
    echo "[LOG] ${log_file}" | tee -a "${RUN_ALL_LOG}"
    echo "----------------------------------------" | tee -a "${RUN_ALL_LOG}"

    "$@" 2>&1 | tee "${log_file}"
    local exit_code=${PIPESTATUS[0]}

    if [[ ${exit_code} -ne 0 ]]; then
        echo "[FAIL] exit_code=${exit_code}" | tee -a "${RUN_ALL_LOG}"
        return ${exit_code}
    fi

    echo "[OK] $*" | tee -a "${RUN_ALL_LOG}"
}

for f in \
    "${TRAIN_FILE}" \
    "${EVAL_FILE}" \
    "${TEST_FILE}" \
    "${INFO_FILE}" \
    "${SID_INDEX_PATH}" \
    "${ITEM_META_PATH}" \
    "./sft.py" \
    "./rl.py" \
    "./collect_results.py" \
    "./extract_log_metrics.py"
do
    if [[ ! -f "${f}" ]]; then
        echo "[FATAL] File not found: ${f}"
        exit 1
    fi
done

echo "RUN_ID=${RUN_ID}" | tee -a "${RUN_ALL_LOG}"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}" | tee -a "${RUN_ALL_LOG}"
echo "NUM_GPUS=${NUM_GPUS}" | tee -a "${RUN_ALL_LOG}"
echo "MASTER_PORT=${MASTER_PORT}" | tee -a "${RUN_ALL_LOG}"
echo "SFT_GLOBAL_BATCH=${SFT_GLOBAL_BATCH}" | tee -a "${RUN_ALL_LOG}"
echo "RL_TRAIN_BATCH=${RL_TRAIN_BATCH}" | tee -a "${RUN_ALL_LOG}"
echo "RL_EVAL_BATCH=${RL_EVAL_BATCH}" | tee -a "${RUN_ALL_LOG}"

log "[Stage 0] Light smoke..."
run_and_log \
    "${LOG_DIR}/smoke_cuda_visibility.log" \
    "${PYTHON_BIN}" -c \
"import os, torch; print('CUDA_VISIBLE_DEVICES=', os.environ.get('CUDA_VISIBLE_DEVICES')); print('cuda_available=', torch.cuda.is_available()); print('device_count=', torch.cuda.device_count())"
log "[Stage 0] Light smoke finished."

log "[Stage 1] SFT base..."
SFT_BASE_DIR="${ROOT_DIR}/sft_base"
run_and_log \
    "${LOG_DIR}/sft_base.log" \
    "${TORCHRUN_BIN}" --nproc_per_node "${NUM_GPUS}" --master_port "$((MASTER_PORT+1))" sft.py \
        --base_model ./models/MiniOneRec/Industrial_ckpt \
        --batch_size "${SFT_GLOBAL_BATCH}" \
        --micro_batch_size "${SFT_MICRO_BATCH}" \
        --train_file "${TRAIN_FILE}" \
        --eval_file "${EVAL_FILE}" \
        --output_dir "${SFT_BASE_DIR}" \
        --wandb_project minionerec_sft \
        --wandb_run_name "${CATEGORY}_base_sft_${RUN_ID}" \
        --category "${CATEGORY}" \
        --train_from_scratch False \
        --seed 42 \
        --sid_index_path "${SID_INDEX_PATH}" \
        --item_meta_path "${ITEM_META_PATH}" \
        --freeze_LLM False \
        --use_level_aux False

log "[Stage 2] SFT aux..."
SFT_AUX_DIR="${ROOT_DIR}/sft_aux"
run_and_log \
    "${LOG_DIR}/sft_aux.log" \
    "${TORCHRUN_BIN}" --nproc_per_node "${NUM_GPUS}" --master_port "$((MASTER_PORT+2))" sft.py \
        --base_model ./models/MiniOneRec/Industrial_ckpt \
        --batch_size "${SFT_GLOBAL_BATCH}" \
        --micro_batch_size "${SFT_MICRO_BATCH}" \
        --train_file "${TRAIN_FILE}" \
        --eval_file "${EVAL_FILE}" \
        --output_dir "${SFT_AUX_DIR}" \
        --wandb_project minionerec_sft \
        --wandb_run_name "${CATEGORY}_aux_sft_${RUN_ID}" \
        --category "${CATEGORY}" \
        --train_from_scratch False \
        --seed 42 \
        --sid_index_path "${SID_INDEX_PATH}" \
        --item_meta_path "${ITEM_META_PATH}" \
        --freeze_LLM False \
        --use_level_aux True \
        --level_weight_0 0.1 \
        --level_weight_1 0.1 \
        --level_weight_2 0.1

log "[Stage 3] RL base..."
RL_BASE_DIR="${ROOT_DIR}/rl_base"
run_and_log \
    "${LOG_DIR}/rl_base.log" \
    "${ACCELERATE_BIN}" launch --num_processes "${NUM_GPUS}" --main_process_port "$((MASTER_PORT+3))" rl.py \
        --model_path "${SFT_BASE_DIR}/final_checkpoint" \
        --train_file "${TRAIN_FILE}" \
        --eval_file "${EVAL_FILE}" \
        --info_file "${INFO_FILE}" \
        --output_dir "${RL_BASE_DIR}" \
        --wandb_project minionerec_rl \
        --wandb_run_name "${CATEGORY}_base_rl_${RUN_ID}" \
        --category "${CATEGORY}" \
        --seed 42 \
        --sid_index_path "${SID_INDEX_PATH}" \
        --item_meta_path "${ITEM_META_PATH}" \
        --train_batch_size "${RL_TRAIN_BATCH}" \
        --eval_batch_size "${RL_EVAL_BATCH}" \
        --num_train_epochs "${RL_NUM_EPOCHS}" \
        --gradient_accumulation_steps "${RL_GRAD_ACC}" \
        --num_generations "${RL_NUM_GENERATIONS}" \
        --beam_search "${RL_BEAM_SEARCH}"

log "[Stage 4] RL aux..."
RL_AUX_DIR="${ROOT_DIR}/rl_aux"
run_and_log \
    "${LOG_DIR}/rl_aux.log" \
    "${ACCELERATE_BIN}" launch --num_processes "${NUM_GPUS}" --main_process_port "$((MASTER_PORT+4))" rl.py \
        --model_path "${SFT_AUX_DIR}/final_checkpoint" \
        --train_file "${TRAIN_FILE}" \
        --eval_file "${EVAL_FILE}" \
        --info_file "${INFO_FILE}" \
        --output_dir "${RL_AUX_DIR}" \
        --wandb_project minionerec_rl \
        --wandb_run_name "${CATEGORY}_aux_rl_${RUN_ID}" \
        --category "${CATEGORY}" \
        --seed 42 \
        --sid_index_path "${SID_INDEX_PATH}" \
        --item_meta_path "${ITEM_META_PATH}" \
        --train_batch_size "${RL_TRAIN_BATCH}" \
        --eval_batch_size "${RL_EVAL_BATCH}" \
        --num_train_epochs "${RL_NUM_EPOCHS}" \
        --gradient_accumulation_steps "${RL_GRAD_ACC}" \
        --num_generations "${RL_NUM_GENERATIONS}" \
        --beam_search "${RL_BEAM_SEARCH}"

log "[Stage 5] Collect results..."
run_and_log \
    "${LOG_DIR}/collect_results.log" \
    "${PYTHON_BIN}" collect_results.py \
        --root_dir "${ROOT_DIR}" \
        --output_csv "${CSV_DIR}/all_results.csv"

run_and_log \
    "${LOG_DIR}/extract_log_metrics.log" \
    "${PYTHON_BIN}" extract_log_metrics.py \
        --log_dir "${LOG_DIR}" \
        --output_csv "${CSV_DIR}/all_log_metrics.csv"

log "All experiments finished."
log "ROOT_DIR=${ROOT_DIR}"
log "Results CSV: ${CSV_DIR}/all_results.csv"
log "Log metrics CSV: ${CSV_DIR}/all_log_metrics.csv"