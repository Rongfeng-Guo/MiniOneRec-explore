#!/bin/bash

# 我们需要8张显卡，每张卡至少需要 35000 MB (约35GB) 的空闲显存才算“空闲”
REQUIRED_GPUS=8
MIN_FREE_MEMORY=35000

echo "👀 开始监控 GPU 状态..."
echo "等待目标：至少 $REQUIRED_GPUS 张显卡，且每张卡空闲显存 > $MIN_FREE_MEMORY MB"

while true; do
    # 获取当前所有 GPU 的空闲显存（纯数字，单位MB）
    free_memory_list=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits)
    
    # 统计达标的显卡数量
    available_gpus=0
    for mem in $free_memory_list; do
        if [ "$mem" -ge "$MIN_FREE_MEMORY" ]; then
            ((available_gpus++))
        fi
    done

    # 判断是否满足启动条件
    if [ "$available_gpus" -ge "$REQUIRED_GPUS" ]; then
        echo "=========================================="
        echo "🚀 $(date): 发现 $available_gpus 张显卡空闲！条件满足，立即启动训练！"
        echo "=========================================="
        
        # 启动你的训练脚本
        bash sft.sh
        
        # 跑完后退出监控循环
        echo "🎉 $(date): sft.sh 执行完毕，监控脚本退出。"
        break
    else
        # 终端打印一下当前状态，然后休眠 60 秒再次检查
        echo "⏳ $(date): 当前仅有 $available_gpus 张卡达标，继续等待..."
        sleep 60
    fi
done