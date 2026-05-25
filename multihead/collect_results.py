import os
import re
import json
import ast
import argparse
from statistics import mean


TARGET_KEYS = [
    "loss",
    "grad_norm",
    "learning_rate",
    "reward",
    "reward_std",
    "kl",
    "NDCG@3",
    "HR@3",
    "NDCG@5",
    "HR@5",
    "NDCG@10",
    "HR@10",
    "NDCG@20",
    "HR@20",
    "rewards/rule_reward",
    "rewards/ndcg_rule_reward",
    "categorical_diversity",
    "token_diversity",
    "completion_length",
    "epoch",
]


def safe_float(x):
    try:
        return float(x)
    except Exception:
        return None


def parse_log_file(log_path):
    stats = {k: [] for k in TARGET_KEYS}
    raw_records = []

    if not os.path.exists(log_path):
        return {"exists": False}

    with open(log_path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip()

            if not line:
                continue

            # 只抓像 {'loss': ..., ...} 这种 trainer log
            if line.startswith("{") and line.endswith("}"):
                try:
                    obj = ast.literal_eval(line)
                    if isinstance(obj, dict):
                        raw_records.append(obj)
                        for k in TARGET_KEYS:
                            if k in obj:
                                v = safe_float(obj[k])
                                if v is not None:
                                    stats[k].append(v)
                except Exception:
                    pass

    summary = {
        "exists": True,
        "num_records": len(raw_records),
        "last": {},
        "mean": {},
        "min": {},
        "max": {},
    }

    if raw_records:
        last_rec = raw_records[-1]
        for k in TARGET_KEYS:
            if k in last_rec:
                v = safe_float(last_rec[k])
                if v is not None:
                    summary["last"][k] = v

    for k, vals in stats.items():
        if len(vals) > 0:
            summary["mean"][k] = mean(vals)
            summary["min"][k] = min(vals)
            summary["max"][k] = max(vals)

    # 简单稳定性诊断
    summary["stability"] = {
        "max_kl": summary["max"].get("kl"),
        "max_grad_norm": summary["max"].get("grad_norm"),
        "max_loss": summary["max"].get("loss"),
        "warning_high_kl": (summary["max"].get("kl", 0.0) is not None and summary["max"].get("kl", 0.0) > 3.0),
        "warning_high_grad": (summary["max"].get("grad_norm", 0.0) is not None and summary["max"].get("grad_norm", 0.0) > 10.0),
        "warning_loss_spike": (summary["max"].get("loss", 0.0) is not None and summary["max"].get("loss", 0.0) > 1.0),
    }

    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_root", type=str, required=True)
    args = parser.parse_args()

    run_root = args.run_root
    log_dir = os.path.join(run_root, "logs")

    if not os.path.isdir(log_dir):
        raise FileNotFoundError(f"log_dir not found: {log_dir}")

    for exp_name in sorted(os.listdir(log_dir)):
        if not exp_name.endswith(".log"):
            continue

        log_path = os.path.join(log_dir, exp_name)
        parsed = parse_log_file(log_path)

        # 日志名 S1_base_sft.log -> 输出到对应实验目录
        stem = exp_name[:-4]
        exp_dir = os.path.join(run_root, stem)

        if os.path.isdir(exp_dir):
            out_path = os.path.join(exp_dir, "parsed_log_metrics.json")
        else:
            # smoke / master / collect 等不一定有 exp_dir
            misc_dir = os.path.join(run_root, "summary")
            os.makedirs(misc_dir, exist_ok=True)
            out_path = os.path.join(misc_dir, f"{stem}.parsed_log_metrics.json")

        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(parsed, f, indent=2, ensure_ascii=False)

        print(f"[OK] parsed {log_path} -> {out_path}")


if __name__ == "__main__":
    main()