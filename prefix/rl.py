# rl.py
from datasets import Dataset
from trl import GRPOConfig
import random
import numpy as np
import torch
from data import (
    SidDataset,
    RLTitle2SidDataset,
    RLSeqTitle2SidDataset,
)
from torch.utils.data import ConcatDataset
from transformers import AutoTokenizer
import os
from minionerec_trainer import ReReTrainer
from sasrec import SASRec
from fire import Fire
import pickle
import math
import json

from reward_utils import (
    normalize_sid_text,
    safe_split_sid,
    hierarchical_match_score,
    pclpo_soft_score,
    pclpo_margin_score,
    build_sibling_map,
    build_ndcg_penalties,
)

os.environ["WANDB_MODE"] = "disabled"


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def train(
    # model/data params
    model_path: str = "",
    seed: int = 42,
    train_file: str = "",
    eval_file: str = "",
    info_file: str = "",
    category: str = "",

    # wandb params
    wandb_project: str = "",
    wandb_run_name: str = "",

    # training hyperparams
    output_dir: str = "",
    train_batch_size: int = 32,
    eval_batch_size: int = 32,
    gradient_accumulation_steps: int = 1,
    temperature: float = 1.0,
    add_gt: bool = False,
    eval_step: float = 0.199,
    num_generations: int = 16,
    num_train_epochs: int = 1,
    learning_rate: float = 1e-6,
    beta: float = 0.04,
    max_grad_norm: float = 0.3,
    beam_search: bool = False,
    test_during_training: bool = True,
    dynamic_sampling: bool = False,
    mask_all_zero: bool = False,
    sync_ref_model: bool = False,
    test_beam: int = 20,
    reward_type: str = "rule",
    sample_train: bool = False,
    ada_path: str = "",
    cf_path: str = "",
    sid_index_path: str = "",
    item_meta_path: str = "",
    dapo: bool = False,
    gspo: bool = False,

    # hierarchical reward weights
    hier_reward_l0: float = 0.1,
    hier_reward_l1: float = 0.2,
    hier_reward_l2: float = 0.3,
    hier_reward_exact: float = 1.0,

    # PC-LPO params
    pclpo_parent_depth: int = 2,
    pclpo_same_parent_bonus: float = 0.15,
    pclpo_same_parent_reward: float = 0.25,
    pclpo_wrong_prefix_reward: float = 0.0,
):
    torch.backends.cuda.enable_flash_sdp(False)
    torch.backends.cuda.enable_mem_efficient_sdp(False)
    set_seed(seed)

    print(f"[RL] model_path = {model_path!r}")
    print(f"[RL] train_file = {train_file!r}")
    print(f"[RL] eval_file = {eval_file!r}")
    print(f"[RL] info_file = {info_file!r}")
    print(f"[RL] sid_index_path = {sid_index_path!r}")
    print(f"[RL] item_meta_path = {item_meta_path!r}")
    print(f"[RL] output_dir = {output_dir!r}")
    print(f"[RL] category = {category!r}")
    print(f"[RL] reward_type = {reward_type!r}")

    if not model_path or str(model_path).strip() == "":
        raise ValueError(f"[RL] model_path is empty: {model_path!r}")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"[RL] model_path does not exist: {model_path}")

    if not train_file or not os.path.exists(train_file):
        raise FileNotFoundError(f"[RL] train_file not found: {train_file}")
    if not eval_file or not os.path.exists(eval_file):
        raise FileNotFoundError(f"[RL] eval_file not found: {eval_file}")
    if not info_file or not os.path.exists(info_file):
        raise FileNotFoundError(f"[RL] info_file not found: {info_file}")
    if not sid_index_path or not os.path.exists(sid_index_path):
        raise FileNotFoundError(f"[RL] sid_index_path not found: {sid_index_path}")
    if not item_meta_path or not os.path.exists(item_meta_path):
        raise FileNotFoundError(f"[RL] item_meta_path not found: {item_meta_path}")
    if not output_dir:
        raise ValueError("[RL] output_dir is empty")

    category_dict = {
        "Industrial_and_Scientific": "industrial and scientific items",
        "Office_Products": "office products",
        "Toys_and_Games": "toys and games",
        "Sports": "sports and outdoors",
        "Books": "books",
    }

    if category not in category_dict:
        raise ValueError(f"Unsupported category: {category}")

    with open(info_file, "r", encoding="utf-8") as f:
        info = f.readlines()
        # 第一列是 semantic_id / sid
        item_name = [_.split("\t")[0].strip() for _ in info]
        item2id = {name: i for i, name in enumerate(item_name)}

    sibling_map, parent_to_children, sid_set = build_sibling_map(
        all_sids=item_name,
        parent_depth=pclpo_parent_depth,
    )

    print(f"[RL] total valid sid = {len(sid_set)}")
    print(f"[RL] total parent groups(depth={pclpo_parent_depth}) = {len(parent_to_children)}")

    sample = -1
    train_datasets = []

    train_data1 = SidDataset(train_file, category=category_dict[category], sample=sample)
    train_datasets.append(train_data1)

    train_data2 = RLTitle2SidDataset(
        item_file=item_meta_path,
        index_file=sid_index_path,
        category=category_dict[category],
        sample=sample,
    )
    train_datasets.append(train_data2)

    train_data3 = RLSeqTitle2SidDataset(
        train_file,
        category=category_dict[category],
        sample=10000,
    )
    train_datasets.append(train_data3)

    train_data = ConcatDataset(train_datasets)
    eval_data = SidDataset(eval_file, category=category_dict[category], sample=sample)

    train_dataset = Dataset.from_dict(
        {k: [elm[k] for elm in train_data] for k in train_data[0].keys()}
    )
    train_dataset = train_dataset.shuffle(seed=seed)

    if sample_train and "sft" in model_path:
        train_dataset = train_dataset.select(
            range(int(0.2 * len(train_dataset)), len(train_dataset))
        )

    eval_dataset = Dataset.from_dict(
        {k: [elm[k] for elm in eval_data] for k in eval_data[0].keys()}
    )
    eval_dataset = eval_dataset.shuffle(seed=seed)

    prompt2history = {}
    history2target = {}

    for dataset in train_datasets:
        if hasattr(dataset, "prompt2history"):
            prompt2history.update(dataset.prompt2history)
        if hasattr(dataset, "history2target"):
            history2target.update(dataset.history2target)

    if hasattr(eval_data, "prompt2history"):
        prompt2history.update(eval_data.prompt2history)
    if hasattr(eval_data, "history2target"):
        history2target.update(eval_data.history2target)

    print("train_dataset:", train_dataset)
    print("eval_dataset:", eval_dataset)

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    len_seq = 10
    item_num = len(item_name)
    print(f"item_num: {item_num}")

    model = None
    item_ada_embd = None

    if reward_type == "sasrec":
        if not cf_path or not os.path.exists(cf_path):
            raise FileNotFoundError(f"[RL] cf_path not found for sasrec reward: {cf_path}")
        model = SASRec(32, item_num, len_seq, 0.3, device)
        model.to(device)
        model.load_state_dict(torch.load(cf_path, map_location=device))
        model.eval()

    if reward_type == "semantic":
        if not ada_path or not os.path.exists(ada_path):
            raise FileNotFoundError(f"[RL] ada_path not found for semantic reward: {ada_path}")
        with open(ada_path, "rb") as f:
            item_ada_embd = pickle.load(f)
        item_ada_embd = torch.tensor(item_ada_embd).to(device)

    print("Load reward side resources successfully.")

    ndcg_penalties = build_ndcg_penalties(num_generations=num_generations)

    def get_targets_from_prompts(prompts):
        history = []
        targets = []
        for p in prompts:
            if p not in prompt2history:
                raise KeyError(f"[RL] prompt not found in prompt2history: {repr(p)[:300]}")
            h = prompt2history[p]
            if h not in history2target:
                raise KeyError(f"[RL] history not found in history2target: {repr(h)[:300]}")
            history.append(h)
            targets.append(history2target[h])
        return history, targets

    # ---------------------------
    # 基础 reward
    # ---------------------------
    def rule_reward(prompts, completions):
        _, targets = get_targets_from_prompts(prompts)
        rewards = []
        for i, completion in enumerate(completions):
            pred = normalize_sid_text(completion)
            tgt = normalize_sid_text(targets[i])
            rewards.append(1.0 if pred == tgt else 0.0)
        return rewards

    def ndcg_rule_reward(prompts, completions):
        """
        与原版一致：
        仅当 group 内出现 exact hit 时，才对未命中样本按位置给负奖励；
        若整组全错，则整组 reward=0。
        """
        _, targets = get_targets_from_prompts(prompts)
        rewards = []
        group_rewards = []
        group_has_hit = False

        for i, completion in enumerate(completions):
            pred = normalize_sid_text(completion)
            tgt = normalize_sid_text(targets[i])

            if pred == tgt:
                group_has_hit = True
                group_rewards.append(0.0)
            else:
                group_rewards.append(ndcg_penalties[i % num_generations])

            if (i + 1) % num_generations == 0:
                if group_has_hit:
                    rewards.extend(group_rewards)
                else:
                    rewards.extend([0.0] * num_generations)
                group_rewards = []
                group_has_hit = False

        return rewards

    # ---------------------------
    # hierarchical reward
    # ---------------------------
    def hierarchical_rule_reward(prompts, completions):
        _, targets = get_targets_from_prompts(prompts)
        rewards = []

        for i, completion in enumerate(completions):
            pred = normalize_sid_text(completion)
            tgt = normalize_sid_text(targets[i])

            score = hierarchical_match_score(
                pred_sid=pred,
                target_sid=tgt,
                w_l0=hier_reward_l0,
                w_l1=hier_reward_l1,
                w_l2=hier_reward_l2,
                w_exact=hier_reward_exact,
            )
            rewards.append(score)

        return rewards

    def hierarchical_ndcg_reward(prompts, completions):
        """
        位置加权的 hierarchical reward：
        越靠前位置，hierarchical reward 权重越高
        """
        _, targets = get_targets_from_prompts(prompts)
        rewards = []
        group_rewards = []

        for i, completion in enumerate(completions):
            pred = normalize_sid_text(completion)
            tgt = normalize_sid_text(targets[i])

            base_score = hierarchical_match_score(
                pred_sid=pred,
                target_sid=tgt,
                w_l0=hier_reward_l0,
                w_l1=hier_reward_l1,
                w_l2=hier_reward_l2,
                w_exact=hier_reward_exact,
            )

            rank_bonus = -ndcg_penalties[i % num_generations]  # 正值，前排更大
            group_rewards.append(base_score * rank_bonus)

            if (i + 1) % num_generations == 0:
                rewards.extend(group_rewards)
                group_rewards = []

        return rewards

    # ---------------------------
    # PC-LPO soft
    # ---------------------------
    def pclpo_soft_reward(prompts, completions):
        _, targets = get_targets_from_prompts(prompts)
        rewards = []

        for i, completion in enumerate(completions):
            pred = normalize_sid_text(completion)
            tgt = normalize_sid_text(targets[i])

            score = pclpo_soft_score(
                pred_sid=pred,
                target_sid=tgt,
                sibling_map=sibling_map,
                parent_depth=pclpo_parent_depth,
                w_l0=hier_reward_l0,
                w_l1=hier_reward_l1,
                w_l2=hier_reward_l2,
                w_exact=hier_reward_exact,
                same_parent_bonus=pclpo_same_parent_bonus,
            )
            rewards.append(score)

        return rewards

    def pclpo_soft_ndcg_reward(prompts, completions):
        _, targets = get_targets_from_prompts(prompts)
        rewards = []
        group_rewards = []

        for i, completion in enumerate(completions):
            pred = normalize_sid_text(completion)
            tgt = normalize_sid_text(targets[i])

            base_score = pclpo_soft_score(
                pred_sid=pred,
                target_sid=tgt,
                sibling_map=sibling_map,
                parent_depth=pclpo_parent_depth,
                w_l0=hier_reward_l0,
                w_l1=hier_reward_l1,
                w_l2=hier_reward_l2,
                w_exact=hier_reward_exact,
                same_parent_bonus=pclpo_same_parent_bonus,
            )

            rank_bonus = -ndcg_penalties[i % num_generations]
            group_rewards.append(base_score * rank_bonus)

            if (i + 1) % num_generations == 0:
                rewards.extend(group_rewards)
                group_rewards = []

        return rewards

    # ---------------------------
    # PC-LPO margin
    # ---------------------------
    def pclpo_margin_reward(prompts, completions):
        _, targets = get_targets_from_prompts(prompts)
        rewards = []

        for i, completion in enumerate(completions):
            pred = normalize_sid_text(completion)
            tgt = normalize_sid_text(targets[i])

            score = pclpo_margin_score(
                pred_sid=pred,
                target_sid=tgt,
                sibling_map=sibling_map,
                parent_depth=pclpo_parent_depth,
                w_l0=hier_reward_l0,
                w_l1=hier_reward_l1,
                w_l2=hier_reward_l2,
                exact_reward=hier_reward_exact,
                same_parent_reward=pclpo_same_parent_reward,
                wrong_prefix_reward=pclpo_wrong_prefix_reward,
            )
            rewards.append(score)

        return rewards

    def pclpo_margin_ndcg_reward(prompts, completions):
        _, targets = get_targets_from_prompts(prompts)
        rewards = []
        group_rewards = []

        for i, completion in enumerate(completions):
            pred = normalize_sid_text(completion)
            tgt = normalize_sid_text(targets[i])

            base_score = pclpo_margin_score(
                pred_sid=pred,
                target_sid=tgt,
                sibling_map=sibling_map,
                parent_depth=pclpo_parent_depth,
                w_l0=hier_reward_l0,
                w_l1=hier_reward_l1,
                w_l2=hier_reward_l2,
                exact_reward=hier_reward_exact,
                same_parent_reward=pclpo_same_parent_reward,
                wrong_prefix_reward=pclpo_wrong_prefix_reward,
            )

            rank_bonus = -ndcg_penalties[i % num_generations]
            group_rewards.append(base_score * rank_bonus)

            if (i + 1) % num_generations == 0:
                rewards.extend(group_rewards)
                group_rewards = []

        return rewards

    # ---------------------------
    # semantic reward
    # ---------------------------
    def semantic_reward(prompts, completions):
        _, targets = get_targets_from_prompts(prompts)
        target_ids = [item2id[normalize_sid_text(x)] for x in targets]

        completions = [normalize_sid_text(x) for x in completions]
        completion_ids = []
        for i, elm in enumerate(completions):
            if elm not in item2id:
                print("==============================")
                print(prompts[i])
                print(f"Invalid item: {elm}")
                print("==============================")
                completion_ids.append(random.randint(0, item_num - 1))
            else:
                completion_ids.append(item2id[elm])

        rewards = torch.cosine_similarity(
            item_ada_embd[target_ids],
            item_ada_embd[completion_ids],
            dim=-1,
        )
        return rewards

    # ---------------------------
    # cf / sasrec reward
    # ---------------------------
    def cf_reward(prompts, completions):
        history, _ = get_targets_from_prompts(prompts)
        history_list = [elm.split("::") for elm in history]

        pred_ids = []
        for i, elm in enumerate(completions):
            elm = normalize_sid_text(elm)
            if elm not in item_name:
                pred_ids.append(random.randint(0, item_num - 1))
            else:
                pred_ids.append(item2id[elm])

        len_lis = []
        history_ids = []
        for his in history_list:
            his = [item2id[elm] for elm in his]
            len_lis.append(len(his))
            if len(his) < len_seq:
                his = his + [item_num] * (len_seq - len(his))
            history_ids.append(his)

        seq = torch.LongTensor(history_ids).to(device)
        pred = torch.LongTensor(pred_ids).to(device)

        with torch.no_grad():
            predictions = model.forward_eval(
                seq,
                torch.tensor(np.array(len_lis)).to(device)
            )
            scores = torch.gather(predictions, 1, pred.view(-1, 1)).view(-1)
        return scores

    # ---------------------------
    # reward router
    # ---------------------------
    if reward_type == "rule":
        reward_fun = rule_reward
    elif reward_type == "ranking":
        reward_fun = [rule_reward, ndcg_rule_reward]
    elif reward_type == "ranking_only":
        reward_fun = ndcg_rule_reward

    elif reward_type == "hierarchical":
        reward_fun = hierarchical_rule_reward
    elif reward_type == "hierarchical_ranking":
        reward_fun = [hierarchical_rule_reward, hierarchical_ndcg_reward]
    elif reward_type == "hierarchical_ranking_only":
        reward_fun = hierarchical_ndcg_reward

    elif reward_type == "pclpo_soft":
        reward_fun = pclpo_soft_reward
    elif reward_type == "pclpo_soft_ranking":
        reward_fun = [pclpo_soft_reward, pclpo_soft_ndcg_reward]
    elif reward_type == "pclpo_soft_ranking_only":
        reward_fun = pclpo_soft_ndcg_reward

    elif reward_type == "pclpo_margin":
        reward_fun = pclpo_margin_reward
    elif reward_type == "pclpo_margin_ranking":
        reward_fun = [pclpo_margin_reward, pclpo_margin_ndcg_reward]
    elif reward_type == "pclpo_margin_ranking_only":
        reward_fun = pclpo_margin_ndcg_reward

    elif reward_type == "semantic":
        reward_fun = semantic_reward
    elif reward_type == "sasrec":
        reward_fun = cf_reward
    else:
        raise ValueError(f"Unsupported reward_type: {reward_type}")

    os.environ["WANDB_PROJECT"] = wandb_project
    os.environ["WANDB_MODE"] = "offline"

    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, "exp_config.json"), "w", encoding="utf-8") as f:
        json.dump(
            {
                "model_path": model_path,
                "reward_type": reward_type,
                "learning_rate": learning_rate,
                "beta": beta,
                "max_grad_norm": max_grad_norm,
                "num_generations": num_generations,
                "num_train_epochs": num_train_epochs,
                "train_batch_size": train_batch_size,
                "eval_batch_size": eval_batch_size,
                "gradient_accumulation_steps": gradient_accumulation_steps,
                "hier_reward_l0": hier_reward_l0,
                "hier_reward_l1": hier_reward_l1,
                "hier_reward_l2": hier_reward_l2,
                "hier_reward_exact": hier_reward_exact,
                "pclpo_parent_depth": pclpo_parent_depth,
                "pclpo_same_parent_bonus": pclpo_same_parent_bonus,
                "pclpo_same_parent_reward": pclpo_same_parent_reward,
                "pclpo_wrong_prefix_reward": pclpo_wrong_prefix_reward,
                "beam_search": beam_search,
                "seed": seed,
                "category": category,
                "train_file": train_file,
                "eval_file": eval_file,
                "info_file": info_file,
                "sid_index_path": sid_index_path,
                "item_meta_path": item_meta_path,
            },
            f,
            indent=2,
            ensure_ascii=False,
        )

    training_args = GRPOConfig(
        output_dir=output_dir,
        save_steps=0.1,
        save_total_limit=20,
        eval_strategy="steps",
        max_completion_length=128,
        num_generations=num_generations,
        temperature=temperature,
        sync_ref_model=sync_ref_model,
        per_device_eval_batch_size=eval_batch_size,
        per_device_train_batch_size=train_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        eval_steps=eval_step,
        logging_steps=1,
        learning_rate=learning_rate,
        beta=beta,
        warmup_ratio=0.03,
        max_grad_norm=max_grad_norm,
        num_train_epochs=num_train_epochs,
        bf16=True,
        optim="paged_adamw_32bit",
        lr_scheduler_type="cosine",
        save_strategy="steps",
        report_to="none",
        run_name=wandb_run_name,
    )

    trainer = ReReTrainer(
        model=model_path,
        base_model=model_path,
        dapo=dapo,
        gspo=gspo,
        add_gt=add_gt,
        dynamic_sampling=dynamic_sampling,
        beam_search=beam_search,
        test_during_training=test_during_training,
        test_beam=test_beam,
        info_file=info_file,
        prompt2history=prompt2history,
        history2target=history2target,
        reward_funcs=reward_fun,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        args=training_args,
    )

    trainer.train()
    trainer.save_model(output_dir)

    final_output_dir = os.path.join(output_dir, "final_checkpoint")
    os.makedirs(final_output_dir, exist_ok=True)
    trainer.model.save_pretrained(final_output_dir)
    tokenizer.save_pretrained(final_output_dir)
    print(f"[RL] Saved final checkpoint to: {final_output_dir}")


if __name__ == "__main__":
    Fire(train)