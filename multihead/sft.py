import os
import math
import json
import random
from functools import partial
from typing import Optional, Tuple

import fire
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import transformers
from torch.optim.lr_scheduler import LambdaLR
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    EarlyStoppingCallback,
)

from data import SidSFTDataset


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


def build_sid_level_num_classes(index_path: str):
    """
    从 index.json 构建 3 层 SID 的类别数[K0, K1, K2]
    """
    with open(index_path, "r") as f:
        indices = json.load(f)

    level_sets = [set(), set(), set()]
    for _, sid_tokens in indices.items():
        if not isinstance(sid_tokens, list) or len(sid_tokens) < 3:
            continue
        level_sets[0].add(sid_tokens[0])
        level_sets[1].add(sid_tokens[1])
        level_sets[2].add(sid_tokens[2])

    num_classes_per_level =[len(level_sets[0]), len(level_sets[1]), len(level_sets[2])]
    if min(num_classes_per_level) <= 0:
        raise ValueError(f"Invalid SID class sizes parsed from {index_path}: {num_classes_per_level}")

    return num_classes_per_level


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def _get_cosine_schedule_with_warmup_lr_lambda(
    current_step, *, num_warmup_steps, num_training_steps, num_cycles
):
    if current_step < num_warmup_steps:
        return max(0.1, float(current_step) / float(max(1, num_warmup_steps)))
    progress = float(current_step - num_warmup_steps) / float(
        max(1, num_training_steps - num_warmup_steps)
    )
    return max(0.1, 0.5 * (1.0 + math.cos(math.pi * float(num_cycles) * 2.0 * progress)))


def get_cosine_schedule_with_warmup(
    optimizer, num_warmup_steps, num_training_steps, num_cycles: float = 0.5, last_epoch: int = -1
):
    lr_lambda = partial(
        _get_cosine_schedule_with_warmup_lr_lambda,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps,
        num_cycles=num_cycles,
    )
    return LambdaLR(optimizer, lr_lambda, last_epoch)


class LevelAwareCausalLMWrapper(nn.Module):
    """
    在原始 CausalLM 外面包一层，增加 3 个 SID level 分类头。
    """
    def __init__(self, base_model, hidden_size: int, num_classes_per_level):
        super().__init__()
        self.base_model = base_model
        self.config = base_model.config  
        self.num_classes_per_level = num_classes_per_level

        self.level_head0 = nn.Linear(hidden_size, num_classes_per_level[0])
        self.level_head1 = nn.Linear(hidden_size, num_classes_per_level[1])
        self.level_head2 = nn.Linear(hidden_size, num_classes_per_level[2])

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        labels=None,
        level_pos=None,
        **kwargs,
    ):
        outputs = self.base_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            output_hidden_states=True,
            **kwargs,
        )

        last_hidden = outputs.hidden_states[-1] 

        if level_pos is None:
            batch_size = last_hidden.size(0)
            level_pos = torch.full(
                (batch_size,),
                last_hidden.size(1) - 1,
                dtype=torch.long,
                device=last_hidden.device,
            )
        else:
            level_pos = level_pos.to(last_hidden.device).long()

        batch_indices = torch.arange(last_hidden.size(0), device=last_hidden.device)
        h = last_hidden[batch_indices, level_pos, :]  

        level_logits0 = self.level_head0(h)
        level_logits1 = self.level_head1(h)
        level_logits2 = self.level_head2(h)

        return {
            "loss": outputs.loss,
            "logits": outputs.logits,
            "hidden_states": outputs.hidden_states,
            "past_key_values": getattr(outputs, "past_key_values", None),
            "attentions": getattr(outputs, "attentions", None),
            "level_logits0": level_logits0,
            "level_logits1": level_logits1,
            "level_logits2": level_logits2,
        }

    def get_input_embeddings(self):
        return self.base_model.get_input_embeddings()

    def set_input_embeddings(self, value):
        return self.base_model.set_input_embeddings(value)

    def resize_token_embeddings(self, *args, **kwargs):
        return self.base_model.resize_token_embeddings(*args, **kwargs)

    def save_pretrained(self, save_directory, *args, **kwargs):
        return self.base_model.save_pretrained(save_directory, *args, **kwargs)


class LevelAwareTrainer(transformers.Trainer):
    def __init__(self, *args, level_loss_weights=(0.1, 0.1, 0.1), **kwargs):
        super().__init__(*args, **kwargs)
        self.level_loss_weights = level_loss_weights

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        # 把相关的特有参数都安全地 pop 出来，防止传入不支持 kwargs 的 base model 报错
        target_sid_l0 = inputs.pop("target_sid_l0", None)
        target_sid_l1 = inputs.pop("target_sid_l1", None)
        target_sid_l2 = inputs.pop("target_sid_l2", None)
        level_pos = inputs.pop("level_pos", None)

        # 动态判断模型是否包裹了辅助层 (兼容 DDP 等多卡框架包裹)
        is_level_aware = hasattr(model, "level_head0") or (hasattr(model, "module") and hasattr(model.module, "level_head0"))

        if is_level_aware:
            # === 这里是 SFT AUX 模型的逻辑 ===
            outputs = model(**inputs, level_pos=level_pos)
            
            # Wrapper 返回的是 dict，直接用 dict 提取 loss
            loss_ar = outputs["loss"] if isinstance(outputs, dict) else outputs.loss
            total_loss = loss_ar

            loss_l0 = torch.tensor(0.0, device=loss_ar.device)
            loss_l1 = torch.tensor(0.0, device=loss_ar.device)
            loss_l2 = torch.tensor(0.0, device=loss_ar.device)

            if target_sid_l0 is not None:
                target_sid_l0 = target_sid_l0.to(outputs["level_logits0"].device).long()
                target_sid_l1 = target_sid_l1.to(outputs["level_logits1"].device).long()
                target_sid_l2 = target_sid_l2.to(outputs["level_logits2"].device).long()

                loss_l0 = F.cross_entropy(outputs["level_logits0"], target_sid_l0)
                loss_l1 = F.cross_entropy(outputs["level_logits1"], target_sid_l1)
                loss_l2 = F.cross_entropy(outputs["level_logits2"], target_sid_l2)

                w0, w1, w2 = self.level_loss_weights
                total_loss = loss_ar + w0 * loss_l0 + w1 * loss_l1 + w2 * loss_l2

            self.log(
                {
                    "loss_ar": float(loss_ar.detach().float().item()),
                    "loss_l0": float(loss_l0.detach().float().item()),
                    "loss_l1": float(loss_l1.detach().float().item()),
                    "loss_l2": float(loss_l2.detach().float().item()),
                }
            )

            return (total_loss, outputs) if return_outputs else total_loss

        else:
            # === 这里是 SFT BASE 原生模型的逻辑 ===
            outputs = model(**inputs)
            # 基础模型返回的是 ModelOutput
            loss = outputs.loss if hasattr(outputs, "loss") else outputs["loss"]
            return (loss, outputs) if return_outputs else loss


def train(
    # model/data params
    base_model: str = "",
    train_file: str = "",
    eval_file: str = "",
    output_dir: str = "",
    sample: int = -1,
    seed: int = 42,

    # training hyperparams
    batch_size: int = 128,
    micro_batch_size: int = 4,
    num_epochs: int = 10,
    learning_rate: float = 3e-4,
    cutoff_len: int = 512,

    # llm hyperparams
    group_by_length: bool = False,
    freeze_LLM: bool = False,

    # wandb params
    wandb_project: str = "",
    wandb_run_name: str = "",
    resume_from_checkpoint: str = None,

    category: str = "",
    train_from_scratch: bool = False,
    sid_index_path: str = "",
    item_meta_path: str = "",

    # level-aware aux params
    use_level_aux: bool = False,
    level_weight_0: float = 0.1,
    level_weight_1: float = 0.1,
    level_weight_2: float = 0.1,
):
    set_seed(seed)
    os.environ["WANDB_PROJECT"] = wandb_project

    category_dict = {
        "Industrial_and_Scientific": "industrial and scientific items",
        "Office_Products": "office products",
        "Toys_and_Games": "toys and games",
        "Sports": "sports and outdoors",
        "Books": "books",
    }

    print(f"Original Category: {category}")
    if category in category_dict:
        category = category_dict[category]

    assert base_model, "Please specify --base_model"

    gradient_accumulation_steps = batch_size // micro_batch_size

    world_size = int(os.environ.get("WORLD_SIZE", 1))
    ddp = world_size != 1
    if ddp:
        gradient_accumulation_steps = max(1, gradient_accumulation_steps // world_size)

    if not train_from_scratch:
        model = AutoModelForCausalLM.from_pretrained(
            base_model,
            dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
        )
    else:
        config = AutoConfig.from_pretrained(base_model, trust_remote_code=True)
        model = AutoModelForCausalLM.from_config(config, trust_remote_code=True)
        print("Training from scratch!")

    tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = "left"

    new_tokens =[]
    original_vocab_size = len(tokenizer)

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

        if sid_index_path and os.path.exists(sid_index_path) and len(new_tokens) > 0:
            embedding_layer = model.get_input_embeddings()
            if embedding_layer.weight.shape[0] > original_vocab_size:
                embedding_layer.weight.requires_grad = True

                def mask_grad(grad):
                    grad[:original_vocab_size].zero_()
                    return grad

                embedding_layer.weight.register_hook(mask_grad)

                print(
                    f"Unfrozen {len(new_tokens)} new token embeddings "
                    f"(indices {original_vocab_size} to {len(tokenizer)-1})"
                )
        else:
            print("Warning: freeze_LLM=True but no new tokens added. All parameters are frozen!")

        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in model.parameters())
        print(
            f"Trainable parameters (with grad-mask): {trainable_params:,} / "
            f"{total_params:,} ({100 * trainable_params / total_params:.2f}%)"
        )

    # 载入数据
    train_data = SidSFTDataset(
        train_file=train_file,
        tokenizer=tokenizer,
        max_len=cutoff_len,
        sample=sample,
        seed=seed,
        category=category,
        sid_index_path=sid_index_path,
    )

    val_data = SidSFTDataset(
        train_file=eval_file,
        tokenizer=tokenizer,
        max_len=cutoff_len,
        sample=sample,
        seed=seed,
        category=category,
        sid_index_path=sid_index_path,
    )

    print("LOAD DATA FINISHED")
    print(f"train size: {len(train_data)}")
    print(f"val size: {len(val_data)}")

    if not ddp and torch.cuda.device_count() > 1:
        model.is_parallelizable = True
        model.model_parallel = True

    # 包装 level-aware 头（仅当启用 use_level_aux 时）
    if use_level_aux:
        if not sid_index_path or not os.path.exists(sid_index_path):
            raise ValueError("use_level_aux=True requires a valid --sid_index_path")

        num_classes_per_level = build_sid_level_num_classes(sid_index_path)
        hidden_size = model.config.hidden_size

        model = LevelAwareCausalLMWrapper(
            base_model=model,
            hidden_size=hidden_size,
            num_classes_per_level=num_classes_per_level,
        )
        print(f"Use level-aware aux heads with class sizes: {num_classes_per_level}")
    else:
        print("Running pure BASE model without level-aware auxiliary heads.")

    eval_step = 0.05

    trainer = LevelAwareTrainer(
        model=model,
        train_dataset=train_data,
        eval_dataset=val_data,
        level_loss_weights=(level_weight_0, level_weight_1, level_weight_2),
        args=transformers.TrainingArguments(
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
            eval_steps=eval_step,
            save_strategy="steps",
            save_steps=eval_step,
            output_dir=output_dir,
            save_total_limit=1,
            load_best_model_at_end=True,
            ddp_find_unused_parameters=False if ddp else None,
            report_to="none",
            remove_unused_columns=False,
            group_by_length=group_by_length,
            save_safetensors=False,
        ),
        data_collator=transformers.DataCollatorForSeq2Seq(
            tokenizer,
            pad_to_multiple_of=8,
            return_tensors="pt",
            padding=True,
        ),
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],
    )

    if hasattr(model, "config"):
        model.config.use_cache = False
    elif hasattr(model, "base_model") and hasattr(model.base_model, "config"):
        model.base_model.config.use_cache = False

    # 1. 训练
    trainer.train(resume_from_checkpoint=resume_from_checkpoint)
    
    # 2. 保存中间和最终结果
    trainer.save_model(output_dir)

    final_output_dir = os.path.join(output_dir, "final_checkpoint")
    os.makedirs(final_output_dir, exist_ok=True)

    # 通过 Accelerator 剥离 DDP 得到内部模型
    unwrapped_model = trainer.accelerator.unwrap_model(trainer.model)

    if use_level_aux and hasattr(unwrapped_model, "base_model"):
        # 对于 Aux 模型：只保存它的 base_model 核心部分，以防丢失 config
        unwrapped_model.base_model.save_pretrained(final_output_dir)
        # 额外独立保存 level-aware 的权重（如果你以后还用的话）
        torch.save(
            {
                "level_head0": unwrapped_model.level_head0.state_dict(),
                "level_head1": unwrapped_model.level_head1.state_dict(),
                "level_head2": unwrapped_model.level_head2.state_dict(),
                "num_classes_per_level": unwrapped_model.num_classes_per_level,
            },
            os.path.join(final_output_dir, "level_aux_heads.pt"),
        )
    else:
        # 对于 Base 模型：直接整个保存
        unwrapped_model.save_pretrained(final_output_dir)

    tokenizer.save_pretrained(final_output_dir)
    print(f"Saved final checkpoint to: {final_output_dir}")


if __name__ == "__main__":
    fire.Fire(train)