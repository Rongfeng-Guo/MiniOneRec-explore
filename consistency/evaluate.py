import os
import json
import random

import fire
import numpy as np
import torch
from tqdm import tqdm
from transformers import (
    GenerationConfig,
    AutoTokenizer,
    AutoModelForCausalLM,
    LogitsProcessorList,
)

from data import EvalSidDataset
from LogitProcessor import ConstrainedLogitsProcessor


device = "cuda" if torch.cuda.is_available() else "cpu"


def get_hash(x):
    x = [str(_) for _ in x]
    return "-".join(x)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def main(
    base_model: str = "",
    info_file: str = "",
    category: str = "",
    test_data_path: str = "",
    result_json_data: str = "",
    batch_size: int = 4,
    K: int = 0,
    seed: int = 42,
    length_penalty: float = 0.0,
    max_new_tokens: int = 256,
    num_beams: int = 50,
):
    set_seed(seed)

    category_dict = {
        "Industrial_and_Scientific": "industrial and scientific items",
        "Office_Products": "office products",
        "Toys_and_Games": "toys and games",
        "Sports": "sports and outdoors",
        "Books": "books",
    }
    category_text = category_dict[category]
    print(category_text)

    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        dtype=torch.bfloat16,
        device_map="auto",
        local_files_only=True,
    )
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained(
        base_model,
        local_files_only=True,
    )
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = "left"

    with open(info_file, "r") as f:
        info = f.readlines()
        semantic_ids = [line.split("\t")[0].strip() + "\n" for line in info]

    info_semantic = [f"### Response:\n{_}" for _ in semantic_ids]

    if "llama" in base_model.lower():
        prefixID = [tokenizer(_).input_ids[1:] for _ in info_semantic]
    else:
        prefixID = [tokenizer(_).input_ids for _ in info_semantic]

    prefix_index = 4 if "gpt2" in base_model.lower() else 3

    hash_dict = {}
    for ID in prefixID:
        ID.append(tokenizer.eos_token_id)
        for i in range(prefix_index, len(ID)):
            if i == prefix_index:
                hash_number = get_hash(ID[:i])
            else:
                hash_number = get_hash(ID[prefix_index:i])
            if hash_number not in hash_dict:
                hash_dict[hash_number] = set()
            hash_dict[hash_number].add(ID[i])

    for key in hash_dict.keys():
        hash_dict[key] = list(hash_dict[key])

    def prefix_allowed_tokens_fn(batch_id, input_ids):
        hash_number = get_hash(input_ids)
        if hash_number in hash_dict:
            return hash_dict[hash_number]
        return []

    val_dataset = EvalSidDataset(
        train_file=test_data_path,
        tokenizer=tokenizer,
        max_len=2560,
        category=category_text,
        test=True,
        K=K,
        seed=seed,
    )

    encodings = [val_dataset[i] for i in range(len(val_dataset))]
    test_data = val_dataset.get_all()

    model.config.pad_token_id = tokenizer.eos_token_id
    model.config.eos_token_id = tokenizer.eos_token_id
    model.config.bos_token_id = tokenizer.bos_token_id

    def evaluate_batch(batch_encodings):
        max_len = max(len(x["input_ids"]) for x in batch_encodings)

        padded_input_ids = []
        padded_attention_mask = []

        for x in batch_encodings:
            L = len(x["input_ids"])
            padded_input_ids.append([tokenizer.pad_token_id] * (max_len - L) + x["input_ids"])
            padded_attention_mask.append([0] * (max_len - L) + [1] * L)

        generation_config = GenerationConfig(
            num_beams=num_beams,
            num_return_sequences=num_beams,
            length_penalty=length_penalty,
            pad_token_id=model.config.pad_token_id,
            eos_token_id=model.config.eos_token_id,
            max_new_tokens=max_new_tokens,
            top_k=None,
            top_p=None,
        )

        with torch.no_grad():
            clp = ConstrainedLogitsProcessor(
                prefix_allowed_tokens_fn=prefix_allowed_tokens_fn,
                num_beams=num_beams,
                base_model=base_model,
                eos_token_id=model.config.eos_token_id,
            )
            logits_processor = LogitsProcessorList([clp])

            generation_output = model.generate(
                torch.tensor(padded_input_ids).to(device),
                attention_mask=torch.tensor(padded_attention_mask).to(device),
                generation_config=generation_config,
                return_dict_in_generate=True,
                output_scores=True,
                logits_processor=logits_processor,
            )

        batched_completions = generation_output.sequences[:, max_len:]

        if "llama" in base_model.lower():
            output = tokenizer.batch_decode(
                batched_completions,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )
        else:
            output = tokenizer.batch_decode(
                batched_completions,
                skip_special_tokens=True,
            )

        output = [x.split("Response:\n")[-1].strip() for x in output]
        real_outputs = [output[i * num_beams : (i + 1) * num_beams] for i in range(len(output) // num_beams)]
        return real_outputs

    model = model.to(device)

    outputs = []
    blocks = (len(encodings) + batch_size - 1) // batch_size
    split_encodings = [encodings[i * batch_size : (i + 1) * batch_size] for i in range(blocks)]

    for batch in tqdm(split_encodings):
        outputs.extend(evaluate_batch(batch))

    for i, test in enumerate(test_data):
        test["predict"] = outputs[i]
        if "dedup" in test:
            test.pop("dedup")

    os.makedirs(os.path.dirname(result_json_data), exist_ok=True)
    with open(result_json_data, "w") as f:
        json.dump(test_data, f, indent=4)


if __name__ == "__main__":
    fire.Fire(main)