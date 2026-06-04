# Prefix 分支说明

`prefix/` 是 `MiniOneRec-explore` 中面向奖励设计的一条实验分支，重点研究如何在 RL 阶段利用 SID 的前缀层级结构，提升生成式推荐的最终效果。

## 分支目标

这一分支尝试回答的问题是：

如果模型生成的 SID 在粗粒度层级上已经更接近目标，能否通过 reward shaping 把这种结构优势进一步转化为更好的最终推荐结果？

因此，`prefix/` 的重点不在于改写基础 SFT 架构，而在于重新设计 RL 阶段的奖励信号。

## 主要方法

该分支围绕以下方向开展实验：

- 为 SID 的层级前缀匹配提供 partial credit
- 引入 hierarchical reward，显式区分粗层级与细层级命中
- 设计 ranking-style reward，对高概率但错误的候选施加更明确惩罚
- 比较不同 reward 组合在离线 Top-K 指标上的真实收益

这一路线的核心假设是：粗粒度正确往往比完全精确命中更容易学习，而合理利用这类中间信号，也许能帮助模型更稳定地靠近正确 item。

## 关键文件

- `rl.py`：主要的 reward 设计与 RL 训练逻辑
- `sft.py`：该分支使用的 SFT 训练入口
- `evaluate.py`：离线评测逻辑
- `convert_dataset.py`：数据格式转换
- `compare.md`：实验对比、运行记录和问题整理

## 实验链路

`prefix/` 的完整流程与 MiniOneRec 主线保持一致：

1. 数据预处理
2. 文本编码为 item embedding
3. 构建 SID
4. 转换为 SFT / RL 所需格式
5. 进行 SFT
6. 使用 prefix-aware reward 进行 RL
7. 通过 constrained decoding 做离线评测

常用脚本包括：

- `sft.sh`
- `rl.sh`
- `evaluate.sh`

## 当前观察

从现有实验现象来看，`prefix/` 暴露出一个重要特征：

训练阶段 reward、KL 或过程指标的改善，并不意味着最终离线 Top-K 指标一定改善。

这说明 prefix-aware reward 更容易提升“预测结果在结构上更接近目标”这一性质，但这种结构接近性不会自动转化为正确 item 在最终候选中的更高排名。

## 结果解读建议

阅读 `prefix/` 的实验结果时，建议始终区分以下两类结论：

- 训练过程是否更稳定
- 最终推荐质量是否真的提升

前者不能直接推出后者。这也是这一分支最需要谨慎解释的地方。

## 后续改进建议

如果继续推进 `prefix/`，建议优先关注：

1. 对所有 reward 变体使用统一的离线 Top-K 评测协议
2. 分离分析 coarse prefix 命中率与 exact item 命中率
3. 将关键 reward 参数显式写入脚本与实验记录
4. 在解释 reward 差异前，先确认 constrained decoding 工作正常

## 运行说明

```bash
pip install -r requirements.txt
bash sft.sh
bash rl.sh
bash evaluate.sh
```

如果目标是快速理解这一分支，建议优先阅读 `compare.md` 和 `rl.py`。
