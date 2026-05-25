#!/bin/bash

export NCCL_IB_DISABLE=1
export NCCL_P2P_DISABLE=1
export PYDANTIC_SKIP_VALIDATION=1
export OMP_NUM_THREADS=1

for category in "Industrial_and_Scientific"; do
    train_file=$(ls -f ./data/Amazon/train/${category}*.csv)
    eval_file=$(ls -f ./data/Amazon/valid/${category}*11.csv)
    info_file=$(ls -f ./data/Amazon/info/${category}*.txt)

    echo "🚀 开始正式对齐论文配置进行 RL 训练 (基于多头 Aux-SFT 强化)..."

    # 注意：输出路径已经修改为 ./rl_output_aux，避免覆盖之前的权重
    # 严格使用 8 张卡，对齐论文配置
    CUDA_VISIBLE_DEVICES=2,3,4,5,6,7,8,9 HF_ENDPOINT=https://hf-mirror.com accelerate launch \
        --config_file ./config/zero2_opt.yaml \
        --num_processes 8 --main_process_port 29503 \
        rl.py \
        --model_path ./output/final_checkpoint \
        --train_batch_size 32 \
        --eval_batch_size 64 \
        --num_train_epochs 2 \
        --gradient_accumulation_steps 2 \
        --train_file ${train_file} \
        --eval_file ${eval_file} \
        --info_file ${info_file} \
        --category ${category} \
        --reward_type ranking \
        --num_generations 16 \
        --beam_search True \
        --learning_rate 1e-5 \
        --beta 0.04 \
        --output_dir ./rl_output_aux \
        --sid_index_path ./data/Amazon/index/Industrial_and_Scientific.index.json \
        --item_meta_path ./data/Amazon/index/Industrial_and_Scientific.item.json
done