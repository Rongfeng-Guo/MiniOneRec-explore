#!/bin/bash
# 对比评估两个 SFT 模型

category="Industrial_and_Scientific"

# 1. 评估原版 SFT
echo "Evaluating Baseline SFT..."
# 将下面的路径替换为你原版 SFT 的输出路径
bash evaluate.sh "./output/baseline_sft_checkpoint" $category 

# 2. 评估多头辅助 SFT
echo "Evaluating Aux-Head SFT..."
# 我们昨天跑出来的路径
bash evaluate.sh "./output/final_checkpoint" $category