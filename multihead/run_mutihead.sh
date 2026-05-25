#!/bin/bash
set -Eeuo pipefail

PYTHON_BIN="$(which python)"
ACCELERATE_BIN="$(which accelerate)"

if [[ ! -x "${PYTHON_BIN}" ]]; then
    echo "[FATAL] python not found in current environment"
    exit 1
fi
if [[ ! -x "${ACCELERATE_BIN}" ]]; then
    echo "[FATAL] accelerate not found in current environment"
    exit 1
fi

# 复用你已经跑好的 SFT
BASE_RUN_ID="20260311_154716"

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

SFT_BASE_CKPT="./experiments/runs/${BASE_RUN_ID}/sft_base/final_checkpoint"
SFT_AUX_CKPT="./experiments/runs/${BASE_RUN_ID}/sft_aux/final_checkpoint"

# 排除物理 GPU 6，使用 1,2,3,4,5,7,8,9 共 8 卡
export CUDA_VISIBLE_DEVICES="1,2,3,4,5,7,8,9"
NUM_GPUS=8
MASTER_PORT=29660

# RL 这里必须固定回已验证配置，别再改成 24/48
RL_TRAIN_BATCH=32
RL_EVAL_BATCH=64
RL_NUM_GENERATIONS=16
RL_BEAM_SEARCH=True
RL_NUM_EPOCHS=2
RL_GRAD_ACC=2

export NCCL_IB_DISABLE=1
export NCCL_P2P_DISABLE=1
export PYDANTIC_SKIP_VALIDATION=1
export OMP_NUM_THREADS=1
export TOKENIZERS_PARALLELISM=false
export WANDB_MODE=offline
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONUNBUFFERED=1

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
    "${SFT_BASE_CKPT}/model.safetensors" \
    "${SFT_AUX_CKPT}/model.safetensors" \
    "./rl.py" \
    "./collect_results.py" \
    "./extract_log_metrics.py"
do
    if [[ ! -f "${f}" ]]; then
        echo "[FATAL] File not found: ${f}"
        exit 1
    fi
done

echo "BASE_RUN_ID=${BASE_RUN_ID}" | tee -a "${RUN_ALL_LOG}"
echo "RUN_ID=${RUN_ID}" | tee -a "${RUN_ALL_LOG}"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}" | tee -a "${RUN_ALL_LOG}"
echo "NUM_GPUS=${NUM_GPUS}" | tee -a "${RUN_ALL_LOG}"
echo "MASTER_PORT=${MASTER_PORT}" | tee -a "${RUN_ALL_LOG}"
echo "RL_TRAIN_BATCH=${RL_TRAIN_BATCH}" | tee -a "${RUN_ALL_LOG}"
echo "RL_EVAL_BATCH=${RL_EVAL_BATCH}" | tee -a "${RUN_ALL_LOG}"
echo "SFT_BASE_CKPT=${SFT_BASE_CKPT}" | tee -a "${RUN_ALL_LOG}"
echo "SFT_AUX_CKPT=${SFT_AUX_CKPT}" | tee -a "${RUN_ALL_LOG}"

log "[Stage 0] Light smoke..."
run_and_log \
    "${LOG_DIR}/smoke_cuda_visibility.log" \
    "${PYTHON_BIN}" -c \
"import os, torch; print('CUDA_VISIBLE_DEVICES=', os.environ.get('CUDA_VISIBLE_DEVICES')); print('cuda_available=', torch.cuda.is_available()); print('device_count=', torch.cuda.device_count())"
log "[Stage 0] Light smoke finished."

log "[Stage 1] RL base..."
RL_BASE_DIR="${ROOT_DIR}/rl_base"
run_and_log \
    "${LOG_DIR}/rl_base.log" \
    "${ACCELERATE_BIN}" launch --num_processes "${NUM_GPUS}" --main_process_port "$((MASTER_PORT+1))" rl.py \
        --model_path "${SFT_BASE_CKPT}" \
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

log "[Stage 2] RL aux..."
RL_AUX_DIR="${ROOT_DIR}/rl_aux"
run_and_log \
    "${LOG_DIR}/rl_aux.log" \
    "${ACCELERATE_BIN}" launch --num_processes "${NUM_GPUS}" --main_process_port "$((MASTER_PORT+2))" rl.py \
        --model_path "${SFT_AUX_CKPT}" \
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

log "[Stage 3] Collect results..."
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