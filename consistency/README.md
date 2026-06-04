# Consistency 分支说明

`consistency/` 是 `MiniOneRec-explore` 中面向多视角学习的一条实验分支，重点研究如何通过 SID 视角与文本视角的一致性训练，提升生成式推荐模型的表示稳健性。

## 分支目标

这一分支关注的问题是：

如果使用两种不同输入视角去预测同一个 next SID，并显式约束它们的预测分布和中间表征保持一致，能否学到更稳健、更具泛化性的推荐表示？

这里使用的两个视角分别是：

- `SID history -> next SID`
- `title history -> next SID`

该设计希望把离散 SID 编码的结构信息与自然语言 title 的语义信息结合起来。

## 主要方法

该分支采用 paired multi-view SFT 方案：

1. 为同一条训练样本构造 SID 视角和 title 视角
2. 两个视角分别进行前向计算
3. 两个视角同时优化 next-SID 的 CE loss
4. 额外加入分布一致性与表征一致性约束

需要强调的是，这条线并不是在原始 SFT 方案上附加一个轻量正则，而是引入了一种新的训练样本组织方式和优化路径。

## 关键文件

- `sft_mv.py`：多视角训练主入口
- `paired_mv_dataset.py`：构造 paired two-view 数据
- `mv_consistency_trainer.py`：一致性训练逻辑
- `sft.py`：保留原始单视角 SFT 入口，便于对照

## 与基线的关系

这条分支与原始 MiniOneRec 基线的差异，不只是多了一个 loss：

- 输入形式从单视角变为 paired multi-view
- 训练时需要同时处理两个视角的监督
- 优化目标从单一 CE 扩展为 CE 与一致性约束的联合目标

因此，当前结果更适合被理解为“多视角一致性路线的阶段性实验结论”，而不是对原始基线的严格一处增量替换。

## 当前风险与观察

现有实验显示，这条线至少存在以下风险：

1. 一致性目标可能过强，尤其是当 full-vocab 分布对齐和表征对齐同时启用时
2. 两个视角可能被过度压缩为相似表示，从而削弱原本的互补信息
3. 文本视角虽然带来额外语义信息，但也可能同时引入噪声

从已有结果看，这条分支目前尚未优于对应基线。

## 后续改进建议

如果继续推进 `consistency/`，建议优先尝试：

1. 先恢复一个与原始 mixed-task SFT 严格可比的 base
2. 将一致性约束限制在 SID 相关 token 或更局部的表示空间
3. 对一致性损失做渐进式 warmup
4. 分开分析 CE 改善、一致性程度和最终 Top-K 指标之间的关系

## 运行说明

如果从代码入口开始阅读，优先建议查看：

- `sft_mv.py`
- `paired_mv_dataset.py`
- `mv_consistency_trainer.py`

如需直接运行实验：

```bash
pip install -r requirements.txt
python sft_mv.py
```
