export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1        # 完全禁用 IB/RoCE
# Office_Products, Industrial_and_Scientific
for category in "Industrial_and_Scientific"; do
    train_file=$(ls -f ./data/Amazon/train/${category}*11.csv)
    eval_file=$(ls -f ./data/Amazon/valid/${category}*11.csv)
    test_file=$(ls -f ./data/Amazon/test/${category}*11.csv)
    info_file=$(ls -f ./data/Amazon/info/${category}*.txt)
    echo ${train_file} ${eval_file} ${info_file} ${test_file}
    
    # 这里用 8 张 A100 来跑
    CUDA_VISIBLE_DEVICES=2,3,4,5,6,7,8,9 torchrun --nproc_per_node 8 \
            sft.py \
            --base_model ./models/MiniOneRec/Industrial_ckpt \
            --batch_size 1024 \
            --micro_batch_size 16 \
            --train_file ${train_file} \
            --eval_file ${eval_file} \
            --output_dir ./output \
            --wandb_project minionerec_sft \
            --wandb_run_name industrial_run \
            --category ${category} \
            --train_from_scratch False \
            --seed 42 \
            --sid_index_path ./data/Amazon/index/Industrial_and_Scientific.index.json \
            --item_meta_path ./data/Amazon/index/Industrial_and_Scientific.item.json \
            --freeze_LLM False
done