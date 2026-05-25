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

# =========================================================
# 用法：
#   bash run_pclpo_ablation_resume_4gpu.sh [EXISTING_RUN_ROOT]
#
# 例子：
#   bash run_pclpo_ablation_resume_4gpu.sh ./experiments/runs/20260315_140310_pclpo_ablation
#
# 如果不传，会自动寻找最新的 *_pclpo_ablation 目录
# =========================================================

find_latest_run_root() {
    local latest=""
    latest="$(ls -td ./experiments/runs/*_pclpo_ablation 2>/dev/null | head -n 1 || true)"
    echo "${latest}"
}

if [[ $# -ge 1 && -n "${1:-}" ]]; then
    ROOT_DIR="$1"
elif [[ -n "${EXISTING_RUN_ROOT:-}" ]]; then
    ROOT_DIR="${EXISTING_RUN_ROOT}"
else
    ROOT_DIR="$(find_latest_run_root)"
fi

if [[ -z "${ROOT_DIR:-}" ]]; then
    echo "[FATAL] No existing run root found."
    echo "Please pass one explicitly, e.g.:"
    echo "  bash run_pclpo_ablation_resume_4gpu.sh ./experiments/runs/20260315_140310_pclpo_ablation"
    exit 1
fi

RUN_ID="$(basename "${ROOT_DIR}")"

LOG_DIR="${ROOT_DIR}/logs"
ERR_DIR="${ROOT_DIR}/errors"
CSV_DIR="${ROOT_DIR}/csv"
SUMMARY_DIR="${ROOT_DIR}/summary"
RESULT_DIR="${ROOT_DIR}/results"

mkdir -p "${ROOT_DIR}" "${LOG_DIR}" "${ERR_DIR}" "${CSV_DIR}" "${SUMMARY_DIR}" "${RESULT_DIR}"

CATEGORY="Industrial_and_Scientific"
TRAIN_FILE="./data/Amazon/train/${CATEGORY}_5_2016-10-2018-11.csv"
EVAL_FILE="./data/Amazon/valid/${CATEGORY}_5_2016-10-2018-11.csv"
TEST_FILE="./data/Amazon/test/${CATEGORY}_5_2016-10-2018-11.csv"
INFO_FILE="./data/Amazon/info/${CATEGORY}_5_2016-10-2018-11.txt"
SID_INDEX_PATH="./data/Amazon/index/${CATEGORY}.index.json"
ITEM_META_PATH="./data/Amazon/index/${CATEGORY}.item.json"

# ========= 稳定可用的 SFT 输出 =========
SFT_CKPT="./output/final_checkpoint"

# ========= 使用 GPU 0,1,2,3 共 4 卡 =========
export CUDA_VISIBLE_DEVICES="0,1,2,3"
NUM_GPUS=4

# =========================================================
# 训练 / 评测超参
# 这里先尽量保持与你之前大实验一致，只把卡数降到 4。
# 如果后续还炸，再进一步缩 batch / generations。
# =========================================================
MASTER_PORT_BASE=29750
RL_TRAIN_BATCH=32
RL_EVAL_BATCH=64
RL_NUM_EPOCHS=2
RL_GRAD_ACC=2
RL_NUM_GENERATIONS=8
RL_BEAM_SEARCH=True
RL_LR=1e-5
RL_BETA=0.04

TEST_BATCH_SIZE=8
TEST_NUM_BEAMS=50
TEST_MAX_NEW_TOKENS=256
TEST_LENGTH_PENALTY=0.0

export NCCL_IB_DISABLE=1
export NCCL_P2P_DISABLE=1
export PYDANTIC_SKIP_VALIDATION=1
export OMP_NUM_THREADS=1
export TOKENIZERS_PARALLELISM=false
export WANDB_MODE=offline
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# 为了稳定性，这里保留分布式调试信息，但关闭更激进的同步调试
export TORCH_DISTRIBUTED_DEBUG=DETAIL
# export NCCL_DEBUG=INFO
# export CUDA_LAUNCH_BLOCKING=1

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
    echo "Main Log  : ${RUN_ALL_LOG}" | tee -a "${RUN_ALL_LOG}"
    echo "========================================" | tee -a "${RUN_ALL_LOG}"
    exit 1
}

trap 'die "${LINENO}" "${BASH_COMMAND}" "$?"' ERR

run_and_log() {
    local log_file="$1"
    local err_file="$2"
    shift 2

    mkdir -p "$(dirname "${log_file}")"
    mkdir -p "$(dirname "${err_file}")"

    echo "" | tee -a "${RUN_ALL_LOG}"
    echo "----------------------------------------" | tee -a "${RUN_ALL_LOG}"
    echo "[RUN] $*" | tee -a "${RUN_ALL_LOG}"
    echo "[LOG] ${log_file}" | tee -a "${RUN_ALL_LOG}"
    echo "[ERR] ${err_file}" | tee -a "${RUN_ALL_LOG}"
    echo "----------------------------------------" | tee -a "${RUN_ALL_LOG}"

    "$@" \
        > >(tee -a "${log_file}") \
        2> >(tee -a "${log_file}" "${err_file}" >&2)

    local exit_code=$?

    if [[ ${exit_code} -ne 0 ]]; then
        echo "[FAIL] exit_code=${exit_code}" | tee -a "${RUN_ALL_LOG}"
        echo "[FAIL] stderr saved to ${err_file}" | tee -a "${RUN_ALL_LOG}"
        return ${exit_code}
    fi

    echo "[OK] $*" | tee -a "${RUN_ALL_LOG}"
}

# =========================================================
# 工具函数
# =========================================================

file_nonempty() {
    local f="$1"
    [[ -f "$f" && -s "$f" ]]
}

has_final_checkpoint() {
    local exp_name="$1"
    local ckpt_dir="${ROOT_DIR}/${exp_name}/final_checkpoint"
    [[ -f "${ckpt_dir}/config.json" ]]
}

has_eval_result() {
    local exp_name="$1"
    local out_json="${RESULT_DIR}/${exp_name}_test.json"
    file_nonempty "${out_json}"
}

# 自动寻找最新的中间 checkpoint
find_latest_resume_checkpoint() {
    local exp_dir="$1"
    local latest=""

    latest="$(find "${exp_dir}" -maxdepth 1 -type d -name 'checkpoint-*' 2>/dev/null | sort -V | tail -n 1 || true)"
    if [[ -n "${latest}" ]]; then
        echo "${latest}"
        return 0
    fi

    latest="$(find "${exp_dir}" -maxdepth 1 -type d \( -name 'global_step*' -o -name 'step*' -o -name 'ckpt*' \) 2>/dev/null | sort -V | tail -n 1 || true)"
    if [[ -n "${latest}" ]]; then
        echo "${latest}"
        return 0
    fi

    echo ""
}

# 检测 rl.py 是否支持 resume 参数
detect_resume_flag() {
    local help_text=""
    help_text="$("${PYTHON_BIN}" rl.py --help 2>&1 || true)"

    if echo "${help_text}" | grep -q -- "--resume_from_checkpoint"; then
        echo "--resume_from_checkpoint"
        return 0
    fi
    if echo "${help_text}" | grep -q -- "--resume"; then
        echo "--resume"
        return 0
    fi
    if echo "${help_text}" | grep -q -- "--checkpoint_path"; then
        echo "--checkpoint_path"
        return 0
    fi

    echo ""
}

RESUME_FLAG="$(detect_resume_flag)"

# 必要文件检查
for f in \
    "${TRAIN_FILE}" \
    "${EVAL_FILE}" \
    "${TEST_FILE}" \
    "${INFO_FILE}" \
    "${SID_INDEX_PATH}" \
    "${ITEM_META_PATH}" \
    "${SFT_CKPT}/config.json" \
    "./rl.py" \
    "./reward_utils.py" \
    "./evaluate.py" \
    "./collect_results.py" \
    "./extract_log_metrics.py"
do
    if [[ ! -f "${f}" ]]; then
        echo "[FATAL] File not found: ${f}"
        exit 1
    fi
done

echo "RUN_ID=${RUN_ID}" | tee -a "${RUN_ALL_LOG}"
echo "ROOT_DIR=${ROOT_DIR}" | tee -a "${RUN_ALL_LOG}"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}" | tee -a "${RUN_ALL_LOG}"
echo "NUM_GPUS=${NUM_GPUS}" | tee -a "${RUN_ALL_LOG}"
echo "SFT_CKPT=${SFT_CKPT}" | tee -a "${RUN_ALL_LOG}"
echo "RESUME_FLAG=${RESUME_FLAG:-<not-supported>}" | tee -a "${RUN_ALL_LOG}"

log "[Stage 0] Light smoke..."
run_and_log \
    "${LOG_DIR}/smoke_cuda_visibility.log" \
    "${ERR_DIR}/smoke_cuda_visibility.err.log" \
    "${PYTHON_BIN}" -c \
"import os, torch; print('CUDA_VISIBLE_DEVICES=', os.environ.get('CUDA_VISIBLE_DEVICES')); print('cuda_available=', torch.cuda.is_available()); print('device_count=', torch.cuda.device_count())"
log "[Stage 0] Light smoke finished."

train_one() {
    local exp_name="$1"
    local reward_type="$2"
    local port="$3"
    shift 3

    local out_dir="${ROOT_DIR}/${exp_name}"
    local log_file="${LOG_DIR}/${exp_name}.log"
    local err_file="${ERR_DIR}/${exp_name}.err.log"
    local resume_ckpt=""

    mkdir -p "${out_dir}"

    if has_final_checkpoint "${exp_name}"; then
        log "[Skip Train] ${exp_name} already has final_checkpoint."
        return 0
    fi

    resume_ckpt="$(find_latest_resume_checkpoint "${out_dir}")"

    log "[Train] ${exp_name} / reward_type=${reward_type}"
    if [[ -n "${resume_ckpt}" ]]; then
        log "[Train] found resume checkpoint: ${resume_ckpt}"
    else
        log "[Train] no intermediate checkpoint found, will start this experiment from scratch."
    fi

    local cmd=(
        "${ACCELERATE_BIN}" launch
        --num_processes "${NUM_GPUS}"
        --main_process_port "${port}"
        rl.py
        --model_path "${SFT_CKPT}"
        --train_file "${TRAIN_FILE}"
        --eval_file "${EVAL_FILE}"
        --info_file "${INFO_FILE}"
        --output_dir "${out_dir}"
        --wandb_project minionerec_rl
        --wandb_run_name "${CATEGORY}_${exp_name}_${RUN_ID}"
        --category "${CATEGORY}"
        --seed 42
        --sid_index_path "${SID_INDEX_PATH}"
        --item_meta_path "${ITEM_META_PATH}"
        --train_batch_size "${RL_TRAIN_BATCH}"
        --eval_batch_size "${RL_EVAL_BATCH}"
        --num_train_epochs "${RL_NUM_EPOCHS}"
        --gradient_accumulation_steps "${RL_GRAD_ACC}"
        --num_generations "${RL_NUM_GENERATIONS}"
        --beam_search "${RL_BEAM_SEARCH}"
        --learning_rate "${RL_LR}"
        --beta "${RL_BETA}"
        --reward_type "${reward_type}"
    )

    if [[ -n "${resume_ckpt}" && -n "${RESUME_FLAG}" ]]; then
        cmd+=("${RESUME_FLAG}" "${resume_ckpt}")
    elif [[ -n "${resume_ckpt}" && -z "${RESUME_FLAG}" ]]; then
        log "[Warn] Found resume checkpoint, but rl.py help does not show a known resume flag."
        log "[Warn] This experiment will restart from scratch unless rl.py internally auto-resumes."
    fi

    if [[ $# -gt 0 ]]; then
        cmd+=("$@")
    fi

    run_and_log \
        "${log_file}" \
        "${err_file}" \
        "${cmd[@]}"
}

eval_one() {
    local exp_name="$1"

    local ckpt_dir="${ROOT_DIR}/${exp_name}/final_checkpoint"
    local out_json="${RESULT_DIR}/${exp_name}_test.json"
    local log_file="${LOG_DIR}/${exp_name}_test.log"
    local err_file="${ERR_DIR}/${exp_name}_test.err.log"

    if has_eval_result "${exp_name}"; then
        log "[Skip Eval] ${exp_name} already has test result: ${out_json}"
        return 0
    fi

    if [[ ! -f "${ckpt_dir}/config.json" ]]; then
        echo "[FATAL] final checkpoint missing: ${ckpt_dir}"
        exit 1
    fi

    log "[Eval] ${exp_name}"

    # 评测放到单卡 0；训练结束后才执行，不与训练并发
    run_and_log \
        "${log_file}" \
        "${err_file}" \
        env CUDA_VISIBLE_DEVICES=0 \
        "${PYTHON_BIN}" evaluate.py \
        --base_model "${ckpt_dir}" \
        --info_file "${INFO_FILE}" \
        --category "${CATEGORY}" \
        --test_data_path "${TEST_FILE}" \
        --result_json_data "${out_json}" \
        --batch_size "${TEST_BATCH_SIZE}" \
        --num_beams "${TEST_NUM_BEAMS}" \
        --max_new_tokens "${TEST_MAX_NEW_TOKENS}" \
        --length_penalty "${TEST_LENGTH_PENALTY}"
}

run_exp() {
    local exp_name="$1"
    local reward_type="$2"
    local port="$3"
    shift 3

    train_one "${exp_name}" "${reward_type}" "${port}" "$@"
    eval_one "${exp_name}"
}

# =========================================================
# 只继续跑 03 / 04 / 05
# 已完成自动跳过
# 未完成且有 checkpoint 则尝试续跑
# =========================================================

run_exp \
    "03_hierarchical_ranking" \
    "hierarchical_ranking" \
    "$((MASTER_PORT_BASE+3))" \
    --hier_reward_l0 0.1 \
    --hier_reward_l1 0.2 \
    --hier_reward_l2 0.3 \
    --hier_reward_exact 1.0

run_exp \
    "04_pclpo_soft" \
    "pclpo_soft" \
    "$((MASTER_PORT_BASE+4))" \
    --hier_reward_l0 0.1 \
    --hier_reward_l1 0.2 \
    --hier_reward_l2 0.3 \
    --hier_reward_exact 1.0 \
    --pclpo_parent_depth 2 \
    --pclpo_same_parent_bonus 0.15

run_exp \
    "05_pclpo_soft_ranking" \
    "pclpo_soft_ranking" \
    "$((MASTER_PORT_BASE+5))" \
    --hier_reward_l0 0.1 \
    --hier_reward_l1 0.2 \
    --hier_reward_l2 0.3 \
    --hier_reward_exact 1.0 \
    --pclpo_parent_depth 2 \
    --pclpo_same_parent_bonus 0.15

# =========================
# 日志提取
# =========================
log "[Summary] collect_results..."
run_and_log \
    "${LOG_DIR}/collect_results.log" \
    "${ERR_DIR}/collect_results.err.log" \
    "${PYTHON_BIN}" collect_results.py \
    --run_root "${ROOT_DIR}"

log "[Summary] extract_log_metrics..."
run_and_log \
    "${LOG_DIR}/extract_log_metrics.log" \
    "${ERR_DIR}/extract_log_metrics.err.log" \
    "${PYTHON_BIN}" extract_log_metrics.py \
    --run_root "${ROOT_DIR}"

# =========================
# 汇总 train/test 结果
# 不再依赖 data.py 里的 split_sid_tokens_from_string
# =========================
log "[Summary] build csv summary..."
export RUN_ROOT_FOR_SUMMARY="${ROOT_DIR}"

"${PYTHON_BIN}" - <<'PY'
import os
import re
import json
import csv

run_root = os.environ.get("RUN_ROOT_FOR_SUMMARY")
if run_root is None or run_root == "":
    raise RuntimeError("RUN_ROOT_FOR_SUMMARY is empty")

exps = [
    "03_hierarchical_ranking",
    "04_pclpo_soft",
    "05_pclpo_soft_ranking",
]

csv_dir = os.path.join(run_root, "csv")
result_dir = os.path.join(run_root, "results")
os.makedirs(csv_dir, exist_ok=True)

sid_re = re.compile(r"<a_(\d+)><b_(\d+)><c_(\d+)>")

def norm(x):
    return str(x).strip().strip('"').strip()

def get_pred(x):
    if isinstance(x, list):
        return norm(x[0]) if x else ""
    return norm(x)

def safe_split_sid(x):
    m = sid_re.fullmatch(norm(x))
    if not m:
        return None
    return m.group(1), m.group(2), m.group(3)

def hier_score(pred, tgt, w0=0.1, w1=0.2, w2=0.3, w_exact=1.0):
    ps = safe_split_sid(pred)
    ts = safe_split_sid(tgt)
    if ps is None or ts is None:
        return 0.0
    s = 0.0
    if ps[0] == ts[0]:
        s += w0
    if ps[1] == ts[1]:
        s += w1
    if ps[2] == ts[2]:
        s += w2
    if norm(pred) == norm(tgt):
        s += w_exact
    return s

# train summary
train_rows = []
for exp in exps:
    path = os.path.join(run_root, exp, "parsed_log_metrics.json")
    row = {"exp": exp}
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            d = json.load(f)
        row["num_records"] = d.get("num_records", "")
        row["mean_loss"] = d.get("mean", {}).get("loss", "")
        row["mean_reward"] = d.get("mean", {}).get("reward", "")
        row["mean_kl"] = d.get("mean", {}).get("kl", "")
        row["mean_NDCG@10"] = d.get("mean", {}).get("NDCG@10", "")
        row["mean_HR@10"] = d.get("mean", {}).get("HR@10", "")
        row["mean_NDCG@20"] = d.get("mean", {}).get("NDCG@20", "")
        row["mean_HR@20"] = d.get("mean", {}).get("HR@20", "")
        row["mean_token_diversity"] = d.get("mean", {}).get("token_diversity", "")
        row["max_grad_norm"] = d.get("max", {}).get("grad_norm", "")
        row["max_kl"] = d.get("max", {}).get("kl", "")
        row["warning_high_grad"] = d.get("stability", {}).get("warning_high_grad", "")
        row["warning_high_kl"] = d.get("stability", {}).get("warning_high_kl", "")
    train_rows.append(row)

train_csv = os.path.join(csv_dir, "train_metric_summary_030405.csv")
with open(train_csv, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=list(train_rows[0].keys()))
    writer.writeheader()
    writer.writerows(train_rows)

# test summary
test_rows = []
for exp in exps:
    path = os.path.join(result_dir, f"{exp}_test.json")
    row = {"exp": exp}
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        n = len(data)
        exact = 0
        valid = 0
        l0 = 0
        l1 = 0
        l2 = 0
        hs = 0.0

        for item in data:
            tgt = norm(item["output"])
            pred = get_pred(item["predict"])

            pt = safe_split_sid(pred)
            tt = safe_split_sid(tgt)

            if pred == tgt:
                exact += 1
            if pt is not None:
                valid += 1
            if pt is not None and tt is not None:
                if pt[0] == tt[0]:
                    l0 += 1
                if pt[1] == tt[1]:
                    l1 += 1
                if pt[2] == tt[2]:
                    l2 += 1

            hs += hier_score(pred, tgt)

        row["num_samples"] = n
        row["exact_match_acc"] = exact / n if n else 0.0
        row["valid_sid_rate"] = valid / n if n else 0.0
        row["level0_acc"] = l0 / n if n else 0.0
        row["level1_acc"] = l1 / n if n else 0.0
        row["level2_acc"] = l2 / n if n else 0.0
        row["mean_hier_score"] = hs / n if n else 0.0
        row["num_exact_match"] = exact
    test_rows.append(row)

test_csv = os.path.join(csv_dir, "test_metric_summary_030405.csv")
with open(test_csv, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=list(test_rows[0].keys()))
    writer.writeheader()
    writer.writerows(test_rows)

print(f"[OK] train summary -> {train_csv}")
print(f"[OK] test summary  -> {test_csv}")
PY

log "All experiments finished."
log "ROOT_DIR=${ROOT_DIR}"
log "Train summary: ${CSV_DIR}/train_metric_summary_030405.csv"
log "Test summary : ${CSV_DIR}/test_metric_summary_030405.csv"