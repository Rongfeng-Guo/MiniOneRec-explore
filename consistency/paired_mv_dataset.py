import ast
import json
import random
from typing import Dict, List, Optional

import pandas as pd
import torch
from torch.utils.data import Dataset
from tqdm import tqdm


class Tokenizer:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.bos_id: Optional[int] = tokenizer.bos_token_id
        self.eos_id: Optional[int] = tokenizer.eos_token_id

    def encode(self, s: str, bos: bool, eos: bool) -> List[int]:
        if not isinstance(s, str):
            raise TypeError(f"Expected string, got {type(s)}")
        tokens = self.tokenizer.encode(s, add_special_tokens=False)
        if bos and self.bos_id is not None:
            tokens = [self.bos_id] + tokens
        if eos and self.eos_id is not None:
            tokens = tokens + [self.eos_id]
        return tokens

    def decode(self, t: List[int]) -> str:
        return self.tokenizer.decode(t)


class PairedSidTitleSFTDataset(Dataset):
    """
    Build two aligned views for the same training row:
    1) SID history -> next SID
    2) Title history -> next SID

    Each sample returns both branches so the trainer can enforce
    cross-view output consistency.
    """

    def __init__(
        self,
        train_file: str,
        index_file: str,
        tokenizer,
        max_len: int = 2048,
        sample: int = -1,
        seed: int = 42,
        category: str = "",
        test: bool = False,
        dedup: bool = False,
    ):
        super().__init__()
        self.data = pd.read_csv(train_file)
        if sample > 0:
            self.data = self.data.sample(sample, random_state=seed).reset_index(drop=True)
        self.tokenizer = Tokenizer(tokenizer)
        self.max_len = max_len
        self.test = test
        self.category = category
        self.dedup = dedup
        random.seed(seed)

        with open(index_file, "r") as f:
            self.indices = json.load(f)

        self.id2sid = {}
        for item_id, sids in self.indices.items():
            if isinstance(sids, list) and len(sids) >= 3:
                self.id2sid[str(item_id)] = "".join(sids[:3])

        self.sid_instruction = (
            "Below is an instruction that describes a task, paired with an input that provides further context. "
            "Write a response that appropriately completes the request.\n\n"
            "### Instruction:\n"
            "Can you predict the next possible item that the user may expect?\n\n"
        )
        self.title_instruction = (
            "Below is an instruction that describes a task, paired with an input that provides further context. "
            "Write a response that appropriately completes the request.\n\n"
            "### Instruction:\n"
            "Based on the user's historical interaction with item titles, predict the semantic ID of the next item they may expect.\n\n"
        )

        self.inputs = []
        self.get_inputs()

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        return self.inputs[idx]

    def get_inputs(self):
        self.inputs = []
        for i in tqdm(range(len(self.data)), desc="Building paired MV dataset"):
            sample = self.pre(i)
            if sample is not None:
                self.inputs.append(sample)

    @staticmethod
    def _safe_eval_list(value):
        if isinstance(value, list):
            return value
        if value is None:
            return []
        if isinstance(value, str):
            try:
                parsed = ast.literal_eval(value)
                return parsed if isinstance(parsed, list) else []
            except Exception:
                return []
        return []

    def _generate_prompt(self, input_text: str) -> str:
        return f"### User Input: \n{input_text}\n\n### Response:\n"

    def _build_lm_branch(self, instruction: str, input_text: str, target_text: str) -> Dict[str, List[int]]:
        prompt_ids = self.tokenizer.encode(instruction, bos=True, eos=False)
        prompt_ids += self.tokenizer.encode(self._generate_prompt(input_text), bos=False, eos=False)

        if self.test:
            tokens = prompt_ids[-self.max_len :]
            return {
                "input_ids": tokens,
                "attention_mask": [1] * len(tokens),
                "labels": [-100] * len(tokens),
                "pred_positions": [-1, -1, -1],
                "anchor_pos": -1,
            }

        target_ids = self.tokenizer.encode(target_text, bos=False, eos=True)
        full_input = prompt_ids + target_ids
        labels = [-100] * len(prompt_ids) + target_ids

        # first 3 target tokens are expected to be the 3 SID tokens.
        first_target_pos = len(prompt_ids)
        target_token_count = min(3, len(target_ids))
        full_pred_positions = [first_target_pos + i - 1 for i in range(target_token_count)]
        while len(full_pred_positions) < 3:
            full_pred_positions.append(-1)
        anchor_pos = first_target_pos - 1

        offset = max(0, len(full_input) - self.max_len)
        input_ids = full_input[-self.max_len :]
        attention_mask = [1] * len(input_ids)
        labels = labels[-self.max_len :]

        pred_positions = []
        for pos in full_pred_positions:
            if pos < 0:
                pred_positions.append(-1)
            else:
                new_pos = pos - offset
                pred_positions.append(new_pos if 0 <= new_pos < len(input_ids) else -1)

        anchor_pos = anchor_pos - offset
        if not (0 <= anchor_pos < len(input_ids)):
            anchor_pos = -1

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
            "pred_positions": pred_positions,
            "anchor_pos": anchor_pos,
        }

    def _get_sid_view(self, row) -> Dict[str, str]:
        history_item_sid = self._safe_eval_list(row.get("history_item_sid", []))
        history = ", ".join(map(str, history_item_sid))
        target_sid = str(row.get("item_sid", ""))
        target_sid = target_sid if target_sid else self.id2sid.get(str(row.get("item_id", "")), "")
        if not target_sid:
            return {}

        last_history_sid = history_item_sid[-1] if history_item_sid else None
        if self.dedup and last_history_sid == target_sid:
            return {}

        return {
            "input": (
                f"The user has interacted with items {history} in chronological order. "
                f"Can you predict the next possible item that the user may expect?"
            ),
            "target_sid": target_sid,
        }

    def _get_title_view(self, row) -> Dict[str, str]:
        history_titles = self._safe_eval_list(row.get("history_item_title", []))
        history = ", ".join([f'"{title}"' for title in history_titles])
        target_sid = self.id2sid.get(str(row.get("item_id", "")), "")
        if not target_sid:
            target_sid = str(row.get("item_sid", ""))
        if not target_sid:
            return {}

        history_item_ids = self._safe_eval_list(row.get("history_item_id", []))
        last_history_item_id = str(history_item_ids[-1]) if history_item_ids else None
        if self.dedup and last_history_item_id == str(row.get("item_id", "")):
            return {}

        return {
            "input": (
                f"The user has interacted with the following {self.category} items in chronological order: {history}. "
                f"Can you predict the next item the user may expect?"
            ),
            "target_sid": target_sid,
        }

    def pre(self, idx: int):
        row = self.data.iloc[idx]
        sid_view = self._get_sid_view(row)
        title_view = self._get_title_view(row)
        if not sid_view or not title_view:
            return None

        sid_branch = self._build_lm_branch(
            instruction=self.sid_instruction,
            input_text=sid_view["input"],
            target_text=sid_view["target_sid"] + "\n",
        )
        title_branch = self._build_lm_branch(
            instruction=self.title_instruction,
            input_text=title_view["input"],
            target_text=title_view["target_sid"] + "\n",
        )

        return {
            "sid_input_ids": sid_branch["input_ids"],
            "sid_attention_mask": sid_branch["attention_mask"],
            "sid_labels": sid_branch["labels"],
            "sid_pred_positions": sid_branch["pred_positions"],
            "sid_anchor_pos": sid_branch["anchor_pos"],
            "title_input_ids": title_branch["input_ids"],
            "title_attention_mask": title_branch["attention_mask"],
            "title_labels": title_branch["labels"],
            "title_pred_positions": title_branch["pred_positions"],
            "title_anchor_pos": title_branch["anchor_pos"],
            "target_sid": sid_view["target_sid"],
        }


class PairedDataCollator:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id

    def _pad_branch(self, features: List[Dict], prefix: str) -> Dict[str, torch.Tensor]:
        input_key = f"{prefix}_input_ids"
        mask_key = f"{prefix}_attention_mask"
        label_key = f"{prefix}_labels"
        pos_key = f"{prefix}_pred_positions"
        anchor_key = f"{prefix}_anchor_pos"

        max_len = max(len(f[input_key]) for f in features)
        input_ids, attention_mask, labels = [], [], []
        pred_positions, anchor_positions = [], []

        for f in features:
            cur_len = len(f[input_key])
            pad_len = max_len - cur_len
            input_ids.append(f[input_key] + [self.pad_token_id] * pad_len)
            attention_mask.append(f[mask_key] + [0] * pad_len)
            labels.append(f[label_key] + [-100] * pad_len)
            pred_positions.append(f[pos_key])
            anchor_positions.append(f[anchor_key])

        return {
            input_key: torch.tensor(input_ids, dtype=torch.long),
            mask_key: torch.tensor(attention_mask, dtype=torch.long),
            label_key: torch.tensor(labels, dtype=torch.long),
            pos_key: torch.tensor(pred_positions, dtype=torch.long),
            anchor_key: torch.tensor(anchor_positions, dtype=torch.long),
        }

    def __call__(self, features: List[Dict]) -> Dict[str, torch.Tensor]:
        sid_batch = self._pad_branch(features, "sid")
        title_batch = self._pad_branch(features, "title")
        batch = {**sid_batch, **title_batch}
        batch["target_sid"] = [f["target_sid"] for f in features]
        return batch
