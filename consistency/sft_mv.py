import json
import math
import os
import random
from functools import partial
from typing import Optional

import fire
import numpy as np
import torch
import transformers
from datasets import Dataset as HFDataset
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import ConcatDataset
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer, EarlyStoppingCallback

from data import FusionSeqRecDataset, SidItemFeatDataset, SidSFTDataset
from mv_consistency_trainer import MultiViewConsistencyTrainer
from paired_mv_dataset import PairedDataCollator, PairedSidTitleSFTDataset


class TokenExtender:
    def __init__(self, data_path, dataset, index_file=".index.json"):
        self.data_path = data_path
        self.dataset = dataset
        self.index_file = index_file
        self.indices = None
        self.new_tokens = None

    def _load_data(self):
        with open(os.path.join(self.data_path, self.dataset + self.index_file), "r") as f:
            self.indices = json.load(f)

    def get_new_tokens(self):
        if self.new_tokens is not None:
            return self.new_tokens
        if self.indices is None:
            self._load_data()

        self.new_tokens = set()
        for index in self.indices.values():
            for token in index:
                self.new_tokens.add(token)
        self.new_tokens = sorted(list(self.new_tokens))
        return self.new_tokens


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def _get_cosine_schedule_with_warmup_lr_lambda(current_step, *, num_warmup_steps, num_training_steps, num_cycles):
    if current_step < num_warmup_steps:
        return max(0.1, float(current_step) / float(max(1, num_warmup_steps)))
    progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
    return max(0.1, 0.5 * (1.0 + math.cos(math.pi * float(num_cycles) * 2.0 * progress)))


def get_cosine_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps, num_cycles: float = 0.5, last_epoch: int = -1):
    lr_lambda = partial(
        _get_cosine_schedule_with_warmup_lr_lambda,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps,
        num_cycles=num_cycles,
    )
    return LambdaLR(optimizer, lr_lambda, last_epoch)


def train(
    base_model: str = "",
    train_file: str = "",
    eval_file: str = "",
    output_dir: str = "",
    sample: int = -1,
    seed: int = 42,
    batch_size: int = 128,
    micro_batch_size: int = 4,
    num_epochs: int = 10,
    learning_rate: float = 3e-4,
    cutoff_len: int = 512,
    group_by_length: bool = False,
    freeze_LLM: bool = False,
    wandb_project: str = "",
    wandb_run_name: str = "",
    resume_from_checkpoint: Optional[str] = None,
    category: str = "",
    train_from_scratch: bool = False,
    sid_index_path: str = "",
    item_meta_path: str = "",
    use_mv_consistency: bool = True,
    use_dist_consistency: bool = True,
    use_repr_consistency: bool = True,
    lambda_dist: float = 0.2,
    lambda_repr: float = 0.05,
    consistency_temperature: float = 1.0,
    no_eval: bool = False,
):
    set_seed(seed)
    os.environ["WANDB_PROJECT"] = wandb_project
    os.environ["WANDB_MODE"] = "disabled"

    category_dict = {
        "Industrial_and_Scientific": "industrial and scientific items",
        "Office_Products": "office products",
        "Toys_and_Games": "toys and games",
        "Sports": "sports and outdoors",
        "Books": "books",
    }
    print(category)
    category_text = category_dict.get(category, category)

    if not base_model:
        raise ValueError("Please specify --base_model")

    gradient_accumulation_steps = batch_size // micro_batch_size
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    ddp = world_size != 1
    if ddp:
        gradient_accumulation_steps = max(1, gradient_accumulation_steps // world_size)

    if not train_from_scratch:
        model = AutoModelForCausalLM.from_pretrained(
            base_model,
            dtype=torch.bfloat16,
            trust_remote_code=True,
            local_files_only=True if os.path.exists(base_model) else False,
        )
    else:
        config = AutoConfig.from_pretrained(
            base_model,
            trust_remote_code=True,
            local_files_only=True if os.path.exists(base_model) else False,
        )
        model = AutoModelForCausalLM.from_config(config, trust_remote_code=True)
        print("Training from scratch!")

    tokenizer = AutoTokenizer.from_pretrained(
        base_model,
        trust_remote_code=True,
        local_files_only=True if os.path.exists(base_model) else False,
    )
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = "left"

    original_vocab_size = len(tokenizer)
    new_tokens = []

    if sid_index_path and os.path.exists(sid_index_path):
        print(f"Loading index from {sid_index_path}")
        token_extender = TokenExtender(
            data_path=os.path.dirname(sid_index_path),
            dataset=os.path.basename(sid_index_path).split(".")[0],
        )
        new_tokens = token_extender.get_new_tokens()
        if new_tokens:
            print(f"Adding {len(new_tokens)} new tokens to tokenizer")
            tokenizer.add_tokens(new_tokens)
            model.resize_token_embeddings(len(tokenizer))

    if freeze_LLM:
        print("Freezing LLM parameters, only training new token embeddings")
        for param in model.parameters():
            param.requires_grad = False

        if new_tokens:
            embedding_layer = model.get_input_embeddings()
            embedding_layer.weight.requires_grad = True

            def mask_grad(grad):
                grad[:original_vocab_size].zero_()
                return grad

            embedding_layer.weight.register_hook(mask_grad)
            print(
                f"Unfrozen {len(new_tokens)} new token embeddings "
                f"(indices {original_vocab_size} to {len(tokenizer) - 1})"
            )
        else:
            print("Warning: freeze_LLM=True but no new tokens were added.")

    if not ddp and torch.cuda.device_count() > 1:
        model.is_parallelizable = True
        model.model_parallel = True

    # -------------------------
    # Dataset
    # -------------------------
    if use_mv_consistency:
        if not sid_index_path:
            raise ValueError("use_mv_consistency=True requires --sid_index_path")

        train_dataset = PairedSidTitleSFTDataset(
            train_file=train_file,
            index_file=sid_index_path,
            tokenizer=tokenizer,
            max_len=cutoff_len,
            sample=sample,
            seed=seed,
            category=category_text,
            dedup=False,
        )

        eval_dataset = None
        if not no_eval:
            eval_dataset = PairedSidTitleSFTDataset(
                train_file=eval_file,
                index_file=sid_index_path,
                tokenizer=tokenizer,
                max_len=cutoff_len,
                sample=sample,
                seed=seed,
                category=category_text,
                dedup=False,
            )

        data_collator = PairedDataCollator(tokenizer)
        trainer_cls = MultiViewConsistencyTrainer
        print("LOAD MV CONSISTENCY DATA FINISHED")
    else:
        train_datasets = []

        train_data1 = SidSFTDataset(
            train_file=train_file,
            tokenizer=tokenizer,
            max_len=cutoff_len,
            sample=sample,
            seed=seed,
            category=category_text,
        )
        train_datasets.append(train_data1)

        train_data2 = SidItemFeatDataset(
            item_file=item_meta_path,
            index_file=sid_index_path,
            tokenizer=tokenizer,
            max_len=cutoff_len,
            sample=sample,
            seed=seed,
            category=category_text,
        )
        train_datasets.append(train_data2)

        train_data3 = FusionSeqRecDataset(
            train_file=train_file,
            item_file=item_meta_path,
            index_file=sid_index_path,
            tokenizer=tokenizer,
            max_len=cutoff_len,
            sample=sample,
            seed=seed,
            category=category_text,
        )
        train_datasets.append(train_data3)

        train_data = ConcatDataset(train_datasets)
        train_dataset = HFDataset.from_dict({k: [v[k] for v in train_data] for k in train_data[0].keys()})
        train_dataset = train_dataset.shuffle(seed=seed)

        eval_dataset = None
        if not no_eval:
            val_data = SidSFTDataset(
                train_file=eval_file,
                tokenizer=tokenizer,
                max_len=cutoff_len,
                sample=sample,
                seed=seed,
                category=category_text,
            )
            eval_dataset = HFDataset.from_dict(
                {k: [v[k] for v in val_data] for k in val_data[0].keys()}
            ).shuffle(seed=seed)

        data_collator = transformers.DataCollatorForSeq2Seq(
            tokenizer,
            pad_to_multiple_of=8,
            return_tensors="pt",
            padding=True,
        )
        trainer_cls = transformers.Trainer
        print("LOAD BASELINE DATA FINISHED")

    print(train_dataset)
    print(eval_dataset)

    # -------------------------
    # TrainingArguments
    # -------------------------
    if no_eval:
        training_args = transformers.TrainingArguments(
            run_name=wandb_run_name,
            per_device_train_batch_size=micro_batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            warmup_steps=20,
            num_train_epochs=num_epochs,
            learning_rate=learning_rate,
            bf16=True,
            logging_steps=1,
            optim="adamw_torch",
            save_strategy="steps",
            save_steps=0.2,
            output_dir=output_dir,
            save_total_limit=1,
            ddp_find_unused_parameters=False if ddp else None,
            report_to="none",
            group_by_length=group_by_length,
            remove_unused_columns=False,
        )
    else:
        training_args = transformers.TrainingArguments(
            run_name=wandb_run_name,
            per_device_train_batch_size=micro_batch_size,
            per_device_eval_batch_size=micro_batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            warmup_steps=20,
            num_train_epochs=num_epochs,
            learning_rate=learning_rate,
            bf16=True,
            logging_steps=1,
            optim="adamw_torch",
            eval_strategy="steps",
            eval_steps=0.5,
            save_strategy="steps",
            save_steps=0.5,
            output_dir=output_dir,
            save_total_limit=1,
            load_best_model_at_end=True,
            ddp_find_unused_parameters=False if ddp else None,
            report_to="none",
            group_by_length=group_by_length,
            remove_unused_columns=False,
        )

    # -------------------------
    # Trainer
    # -------------------------
    callbacks = []
    if not no_eval:
        callbacks.append(EarlyStoppingCallback(early_stopping_patience=3))

    if use_mv_consistency:
        trainer = trainer_cls(
            model=model,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            args=training_args,
            data_collator=data_collator,
            callbacks=callbacks,
            lambda_dist=lambda_dist,
            lambda_repr=lambda_repr,
            consistency_temperature=consistency_temperature,
            use_dist_consistency=use_dist_consistency,
            use_repr_consistency=use_repr_consistency,
        )
    else:
        trainer = trainer_cls(
            model=model,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            args=training_args,
            data_collator=data_collator,
            callbacks=callbacks,
        )

    model.config.use_cache = False

    trainer.train(resume_from_checkpoint=resume_from_checkpoint)
    trainer.save_model(output_dir)

    final_output_dir = os.path.join(output_dir, "final_checkpoint")
    os.makedirs(final_output_dir, exist_ok=True)
    trainer.model.save_pretrained(final_output_dir)
    tokenizer.save_pretrained(final_output_dir)


if __name__ == "__main__":
    fire.Fire(train)