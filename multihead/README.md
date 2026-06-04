# Multihead 分支说明

`multihead/` 是 `MiniOneRec-explore` 中面向层级辅助监督的一条实验分支，重点研究如何在 next-SID 生成任务之外，显式强化模型对 3 层 SID 结构的建模能力。

## 分支目标

这一分支尝试回答的问题是：

如果把目标 SID 拆成 level-0、level-1、level-2 三层标签，并在 SFT 阶段同时优化这些层级预测任务，能否提升模型对层级结构的利用效率，并进一步改善最终推荐结果？

## 主要方法

`multihead/` 的基本做法是：

1. 保留原有 next-SID 自回归生成任务
2. 将目标 SID 拆分为 3 层监督标签
3. 在 SFT 阶段增加 3 个 level classification head
4. 在该 SFT checkpoint 基础上继续进行 RL 训练

相关关键文件包括：

- `data.py`
- `sft.py`
- `rl.py`
- `minionerec_trainer.py`
- `compare.md`

## 与原始基线的关键差异

这一分支最需要注意的地方在于，它并不只是“原始 MiniOneRec + 多头监督”。

当前实现中，至少存在两处会显著影响可比性的变化：

1. 原始 mixed-task SFT 被收窄为单一 `SidSFTDataset`
2. 三个 level head 当前共享同一个 prompt-boundary hidden state

这意味着该分支不仅新增了辅助损失，也改变了基础训练任务和辅助监督的结构位置。

## 方法解释

从训练流程看，`multihead/` 当前的核心逻辑可以概括为：

- 使用 `SidSFTDataset` 训练 next SID 生成
- 从目标 SID 中提取 level-0、level-1、level-2 标签
- 在 AR loss 之外加入 3 个 level CE loss
- 再使用 RL 阶段的 reward 设计继续优化

这一路线的出发点是合理的：如果模型能够先学会较粗层级，再学会较细层级，也许能够更稳定地掌握 SID 结构。

## 当前结果与观察

根据当前最可比的一组本地结果：

| Variant | Samples | Exact@1 | HR@10 | NDCG@10 | HR@20 | NDCG@20 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `rl_base` | 4533 | 0.07831 | 0.15442 | 0.11198 | 0.19347 | 0.12180 |
| `rl_aux` | 4533 | 0.07721 | 0.15773 | 0.11233 | 0.18840 | 0.12008 |

目前可以观察到：

- 辅助监督版本在 `HR@10` 和 `NDCG@10` 上只有轻微波动
- 在 `Exact@1`、`HR@20` 和 `NDCG@20` 上没有形成稳定收益
- 结果更像是对局部样本分布产生了扰动，而不是带来明确整体提升

## 结果为何尚未超过基线

当前较合理的解释包括：

1. SFT 基座已经变化，不再与原始 mixed-task setup 完全一致
2. 三层辅助预测共用一个 hidden state，与真实 SID 自回归生成位置不一致
3. 辅助损失量级足够大时，会明显改写 SFT 阶段的优化目标
4. 粗层级更正确，并不自动等价于最终 item 排名更高

## 后续改进建议

如果继续推进 `multihead/`，当前更值得优先验证的是：

1. 恢复一个与原始 mixed-task SFT 可严格对齐的 base
2. 把 3 个 level head 挂到真实 SID 自回归位置，而不是单一共享 hidden state
3. 让辅助损失逐步 warmup，而不是从训练开始就与 AR loss 等强竞争
4. 统一在离线 Top-K 指标下比较所有变体

## 运行说明

常用入口包括：

- `sft.sh`
- `rl.sh`
- `evaluate.sh`

如果你的目标是先理解这一分支，建议优先阅读：

- `compare.md`
- `sft.py`
- `rl.py`
