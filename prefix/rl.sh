#!/bin/bash

export NCCL_IB_DISABLE=1
export NCCL_P2P_DISABLE=1
export PYDANTIC_SKIP_VALIDATION=1
export OMP_NUM_THREADS=1

for category in "Industrial_and_Scientific"; do
    train_file=$(ls -f ./data/Amazon/train/${category}*.csv)
    eval_file=$(ls -f ./data/Amazon/valid/${category}*11.csv)
    info_file=$(ls -f ./data/Amazon/info/${category}*.txt)

    echo "🚀 开始正式对齐论文配置进行 RL 训练..."

    # 使用 4 张 GPU 对应的实验，逐步训练 02/03/04
    # CUDA_VISIBLE_DEVICES=0,1,2,3 HF_ENDPOINT=https://hf-mirror.com accelerate launch \
    #     --config_file ./config/zero2_opt.yaml \
    #     --num_processes 4 --main_process_port 29503 \
    #     rl.py \
    #     --model_path ./output/final_checkpoint \
    #     --train_batch_size 32 \
    #     --eval_batch_size 64 \
    #     --num_train_epochs 2 \
    #     --gradient_accumulation_steps 2 \
    #     --train_file ${train_file} \
    #     --eval_file ${eval_file} \
    #     --info_file ${info_file} \
    #     --category ${category} \
    #     --reward_type hierarchical \
    #     --num_generations 16 \
    #     --beam_search True \
    #     --learning_rate 1e-5 \
    #     --beta 0.04 \
    #     --output_dir ./rl_output_02_hierarchical \
    #     --sid_index_path ./data/Amazon/index/Industrial_and_Scientific.index.json \
    #     --item_meta_path ./data/Amazon/index/Industrial_and_Scientific.item.json \
    #     --hier_reward_l0 0.1 \
    #     --hier_reward_l1 0.2 \
    #     --hier_reward_l2 0.3 \
    #     --hier_reward_exact 1.0

    # # 训练 03_hierarchical_ranking
    # CUDA_VISIBLE_DEVICES=0,1,2,3 HF_ENDPOINT=https://hf-mirror.com accelerate launch \
    #     --config_file ./config/zero2_opt.yaml \
    #     --num_processes 4 --main_process_port 29503 \
    #     rl.py \
    #     --model_path ./output/final_checkpoint \
    #     --train_batch_size 32 \
    #     --eval_batch_size 64 \
    #     --num_train_epochs 2 \
    #     --gradient_accumulation_steps 2 \
    #     --train_file ${train_file} \
    #     --eval_file ${eval_file} \
    #     --info_file ${info_file} \
    #     --category ${category} \
    #     --reward_type hierarchical_ranking \
    #     --num_generations 16 \
    #     --beam_search True \
    #     --learning_rate 1e-5 \
    #     --beta 0.04 \
    #     --output_dir ./rl_output_03_hierarchical_ranking \
    #     --sid_index_path ./data/Amazon/index/Industrial_and_Scientific.index.json \
    #     --item_meta_path ./data/Amazon/index/Industrial_and_Scientific.item.json \
    #     --hier_reward_l0 0.1 \
    #     --hier_reward_l1 0.2 \
    #     --hier_reward_l2 0.3 \
    #     --hier_reward_exact 1.0

    # 训练 04_pclpo_soft
    CUDA_VISIBLE_DEVICES=4,7,8,9 HF_ENDPOINT=https://hf-mirror.com accelerate launch \
        --config_file ./config/zero2_opt.yaml \
        --num_processes 4 --main_process_port 29503 \
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
        --reward_type pclpo_soft \
        --num_generations 16 \
        --beam_search True \
        --learning_rate 1e-5 \
        --beta 0.04 \
        --output_dir ./rl_output_04_pclpo_soft \
        --sid_index_path ./data/Amazon/index/Industrial_and_Scientific.index.json \
        --item_meta_path ./data/Amazon/index/Industrial_and_Scientific.item.json \
        --hier_reward_l0 0.1 \
        --hier_reward_l1 0.2 \
        --hier_reward_l2 0.3 \
        --hier_reward_exact 1.0 \
        --pclpo_parent_depth 2 \
        --pclpo_same_parent_bonus 0.15


  

    # 训练 05_pclpo_margin_ranking
    CUDA_VISIBLE_DEVICES=4,7,8,9 HF_ENDPOINT=https://hf-mirror.com accelerate launch \
        --config_file ./config/zero2_opt.yaml \
        --num_processes 4 --main_process_port 29505 \
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
        --reward_type pclpo_margin_ranking \
        --num_generations 16 \
        --beam_search True \
        --learning_rate 1e-5 \
        --beta 0.04 \
        --output_dir ./rl_output_06_pclpo_margin_ranking \
        --sid_index_path ./data/Amazon/index/Industrial_and_Scientific.index.json \
        --item_meta_path ./data/Amazon/index/Industrial_and_Scientific.item.json \
        --hier_reward_l0 0.05 \
        --hier_reward_l1 0.10 \
        --hier_reward_l2 0.20 \
        --hier_reward_exact 1.0 \
        --pclpo_parent_depth 2 \
        --pclpo_same_parent_reward 0.15 \
        --pclpo_wrong_prefix_reward 0.0
done