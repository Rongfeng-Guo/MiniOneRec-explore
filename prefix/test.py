import os
import re
import json
from typing import Optional, Tuple, List, Dict

RUN_ROOT = "/root/MiniOneRec-prefix/experiments/runs/20260315_140310_pclpo_ablation"
RESULT_DIR = os.path.join(RUN_ROOT, "results")

EXPS = [
    "01_rule",
    "02_hierarchical",
]

SID_RE = re.compile(r"<a_(\d+)><b_(\d+)><c_(\d+)>")

def norm(x) -> str:
    return str(x).strip().strip('"').strip()

def get_pred(x) -> str:
    if isinstance(x, list):
        return norm(x[0]) if x else ""
    return norm(x)

def parse_sid(x: str) -> Optional[Tuple[str, str, str]]:
    s = norm(x)
    m = SID_RE.fullmatch(s)
    if not m:
        return None
    return m.group(1), m.group(2), m.group(3)

def hierarchical_score(pred: str, tgt: str, w0=0.1, w1=0.2, w2=0.3, w_exact=1.0) -> float:
    ps = parse_sid(pred)
    ts = parse_sid(tgt)
    if ps is None or ts is None:
        return 0.0

    score = 0.0
    if ps[0] == ts[0]:
        score += w0
    if ps[1] == ts[1]:
        score += w1
    if ps[2] == ts[2]:
        score += w2
    if norm(pred) == norm(tgt):
        score += w_exact
    return score

def reciprocal_rank(pred_list: List[str], tgt: str) -> float:
    tgt = norm(tgt)
    for i, p in enumerate(pred_list, start=1):
        if norm(p) == tgt:
            return 1.0 / i
    return 0.0

def hit_at_k(pred_list: List[str], tgt: str, k: int) -> float:
    tgt = norm(tgt)
    topk = [norm(x) for x in pred_list[:k]]
    return 1.0 if tgt in topk else 0.0

def dcg_at_k(pred_list: List[str], tgt: str, k: int) -> float:
    tgt = norm(tgt)
    for i, p in enumerate(pred_list[:k], start=1):
        if norm(p) == tgt:
            # 单一相关项时，IDCG = 1，所以 NDCG = 1/log2(i+1)
            import math
            return 1.0 / math.log2(i + 1)
    return 0.0

def compute_metrics(path: str) -> Dict[str, float]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    n = len(data)
    exact = 0
    valid = 0
    l0 = l1 = l2 = 0
    hs = 0.0

    hr3 = hr5 = hr10 = hr20 = 0.0
    ndcg3 = ndcg5 = ndcg10 = ndcg20 = 0.0
    mrr = 0.0

    for item in data:
        tgt = norm(item["output"])
        preds = item["predict"] if isinstance(item["predict"], list) else [item["predict"]]
        preds = [norm(x) for x in preds]
        pred1 = get_pred(item["predict"])

        pt = parse_sid(pred1)
        tt = parse_sid(tgt)

        if pred1 == tgt:
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

        hs += hierarchical_score(pred1, tgt)

        hr3 += hit_at_k(preds, tgt, 3)
        hr5 += hit_at_k(preds, tgt, 5)
        hr10 += hit_at_k(preds, tgt, 10)
        hr20 += hit_at_k(preds, tgt, 20)

        ndcg3 += dcg_at_k(preds, tgt, 3)
        ndcg5 += dcg_at_k(preds, tgt, 5)
        ndcg10 += dcg_at_k(preds, tgt, 10)
        ndcg20 += dcg_at_k(preds, tgt, 20)

        mrr += reciprocal_rank(preds, tgt)

    return {
        "num_samples": n,
        "exact_match_acc": exact / n if n else 0.0,
        "valid_sid_rate": valid / n if n else 0.0,
        "level0_acc": l0 / n if n else 0.0,
        "level1_acc": l1 / n if n else 0.0,
        "level2_acc": l2 / n if n else 0.0,
        "mean_hier_score": hs / n if n else 0.0,
        "HR@3": hr3 / n if n else 0.0,
        "HR@5": hr5 / n if n else 0.0,
        "HR@10": hr10 / n if n else 0.0,
        "HR@20": hr20 / n if n else 0.0,
        "NDCG@3": ndcg3 / n if n else 0.0,
        "NDCG@5": ndcg5 / n if n else 0.0,
        "NDCG@10": ndcg10 / n if n else 0.0,
        "NDCG@20": ndcg20 / n if n else 0.0,
        "MRR": mrr / n if n else 0.0,
        "num_exact_match": exact,
    }

def main():
    summary = {}
    for exp in EXPS:
        path = os.path.join(RESULT_DIR, f"{exp}_test.json")
        if not os.path.exists(path):
            print(f"[MISS] {path}")
            continue
        metrics = compute_metrics(path)
        summary[exp] = metrics

    print("=" * 80)
    print("FINAL TEST SUMMARY")
    print("=" * 80)
    for exp, metrics in summary.items():
        print(f"\n[{exp}]")
        for k, v in metrics.items():
            print(f"{k:20s}: {v}")

    out_path = os.path.join(RUN_ROOT, "csv", "manual_test_summary.json")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"\n[OK] saved summary to: {out_path}")

if __name__ == "__main__":
    main()