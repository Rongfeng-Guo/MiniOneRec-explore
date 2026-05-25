import math
from collections import defaultdict


def normalize_sid_text(x):
    if x is None:
        return ""
    return str(x).strip().strip('"').strip("'").strip()


def safe_split_sid(x):
    x = normalize_sid_text(x)
    if x == "":
        return None

    # 常见层级分隔写法
    for sep in ["::", "|", ",", "/", "-", "_"]:
        if sep in x:
            parts = [p.strip() for p in x.split(sep) if p.strip() != ""]
            if len(parts) > 0:
                return parts

    # 如果没有显式分隔符，按整体返回
    return [x]


def hierarchical_match_score(
    pred_sid,
    target_sid,
    w_l0=0.1,
    w_l1=0.2,
    w_l2=0.3,
    w_exact=1.0,
):
    pred = safe_split_sid(pred_sid)
    tgt = safe_split_sid(target_sid)

    if pred is None or tgt is None:
        return 0.0

    score = 0.0

    if len(pred) > 0 and len(tgt) > 0 and pred[0] == tgt[0]:
        score += w_l0
    if len(pred) > 1 and len(tgt) > 1 and pred[1] == tgt[1]:
        score += w_l1
    if len(pred) > 2 and len(tgt) > 2 and pred[2] == tgt[2]:
        score += w_l2
    if normalize_sid_text(pred_sid) == normalize_sid_text(target_sid):
        score += w_exact

    return float(score)


def build_sibling_map(all_sids, parent_depth=2):
    sid_set = set(normalize_sid_text(x) for x in all_sids if normalize_sid_text(x) != "")
    parent_to_children = defaultdict(list)

    for sid in sid_set:
        toks = safe_split_sid(sid)
        if toks is None:
            continue
        parent_key = tuple(toks[:parent_depth])
        parent_to_children[parent_key].append(sid)

    sibling_map = {}
    for parent, children in parent_to_children.items():
        child_set = set(children)
        for child in children:
            sibling_map[child] = child_set - {child}

    return sibling_map, dict(parent_to_children), sid_set


def pclpo_soft_score(
    pred_sid,
    target_sid,
    sibling_map,
    parent_depth=2,
    w_l0=0.1,
    w_l1=0.2,
    w_l2=0.3,
    w_exact=1.0,
    same_parent_bonus=0.15,
):
    pred_sid = normalize_sid_text(pred_sid)
    target_sid = normalize_sid_text(target_sid)

    base = hierarchical_match_score(
        pred_sid=pred_sid,
        target_sid=target_sid,
        w_l0=w_l0,
        w_l1=w_l1,
        w_l2=w_l2,
        w_exact=w_exact,
    )

    if pred_sid == target_sid:
        return float(base)

    pred = safe_split_sid(pred_sid)
    tgt = safe_split_sid(target_sid)

    if pred is None or tgt is None:
        return float(base)

    if tuple(pred[:parent_depth]) == tuple(tgt[:parent_depth]):
        base += same_parent_bonus

    return float(base)


def pclpo_margin_score(
    pred_sid,
    target_sid,
    sibling_map,
    parent_depth=2,
    w_l0=0.1,
    w_l1=0.2,
    w_l2=0.3,
    exact_reward=1.0,
    same_parent_reward=0.25,
    wrong_prefix_reward=0.0,
):
    pred_sid = normalize_sid_text(pred_sid)
    target_sid = normalize_sid_text(target_sid)

    if pred_sid == target_sid:
        return float(
            hierarchical_match_score(
                pred_sid,
                target_sid,
                w_l0=w_l0,
                w_l1=w_l1,
                w_l2=w_l2,
                w_exact=exact_reward,
            )
        )

    pred = safe_split_sid(pred_sid)
    tgt = safe_split_sid(target_sid)

    if pred is None or tgt is None:
        return float(wrong_prefix_reward)

    if tuple(pred[:parent_depth]) == tuple(tgt[:parent_depth]):
        return float(same_parent_reward)

    partial = 0.0
    if len(pred) > 0 and len(tgt) > 0 and pred[0] == tgt[0]:
        partial += w_l0
    if len(pred) > 1 and len(tgt) > 1 and pred[1] == tgt[1]:
        partial += w_l1
    if len(pred) > 2 and len(tgt) > 2 and pred[2] == tgt[2]:
        partial += w_l2

    return float(max(partial, wrong_prefix_reward))


def build_ndcg_penalties(num_generations=16):
    penalties = []
    for i in range(num_generations):
        rank = i + 1
        penalties.append(-1.0 / math.log2(rank + 1))
    return penalties