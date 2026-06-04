# MiniOneRec-explore

`MiniOneRec-explore` 是一个围绕 MiniOneRec 展开的研究型代码仓库，用于系统整理 3 条针对 SID 结构增强的实验分支：

- `prefix/`：面向前缀结构的奖励设计
- `multihead/`：面向层级 SID 的辅助监督
- `consistency/`：面向 SID 视角与文本视角的一致性训练

本仓库的目标不是简单汇总实验脚本，而是正式记录这一轮研究探索的设计动机、实现路径、核心观察和当前结论，方便后续复现、对比和继续推进。

## 项目背景

MiniOneRec 的基本思路，是先把每个 item 映射为 3 层 SID，再训练生成式推荐模型根据用户历史预测下一个 SID。基于这一设定，我们进一步关注两个问题：

1. 是否可以更显式地利用 SID 的层级结构？
2. 是否可以把粗到细的结构监督转化为更好的最终推荐质量？

围绕这两个问题，我们分别从奖励函数、辅助监督和多视角学习三个方向开展了实验。

## 项目概览

### `prefix/`

`prefix/` 主要探索 RL 阶段的 reward shaping。该分支尝试通过层级前缀奖励、partial credit 和 ranking-style reward，引导模型不仅关注最终 exact SID 命中，也关注预测结果是否在更粗粒度层面接近目标。

### `multihead/`

`multihead/` 主要探索在 SFT 阶段为 3 层 SID 分别增加辅助分类监督。该分支在 next-SID 自回归目标之外，引入 level-0、level-1、level-2 三个辅助预测头，希望显式强化模型对 SID 层级结构的建模能力。

### `consistency/`

`consistency/` 主要探索 paired multi-view SFT。该分支为同一条样本构造 `SID history -> next SID` 和 `title history -> next SID` 两个视角，并通过分布一致性与表征一致性约束共同训练。

## 当前阶段的核心结论

当前最重要的结论不是“这些结构增强无效”，而是：

这三条实验分支在不同程度上都改变了基础训练问题本身，因此它们与原始 MiniOneRec 基线并非完全等价的“轻量增量对比”。

尤其是在 `multihead/` 和 `consistency/` 中，除了新增 loss 之外，训练任务组织方式、输入构造方式和优化目标都发生了变化。因此，当前实验结果更适合被理解为“结构化增强路线的阶段性研究结论”，而不是最终定型版本。

## 已有结果与观察

目前最可靠、可直接对比的一组本地结果来自同一个 `4533` 样本测试集：

| Variant | Branch | Samples | Exact@1 | HR@10 | NDCG@10 | HR@20 | NDCG@20 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `rl_base` | `multihead` | 4533 | 0.07831 | 0.15442 | 0.11198 | 0.19347 | 0.12180 |
| `rl_aux` | `multihead` | 4533 | 0.07721 | 0.15773 | 0.11233 | 0.18840 | 0.12008 |
| `baseline_sft` | `consistency` | 4533 | 0.07456 | 0.14472 | 0.10479 | 0.18310 | 0.11440 |
| `mv_sft` | `consistency` | 4533 | 0.06684 | 0.13435 | 0.09539 | 0.16854 | 0.10394 |

基于这些结果，目前可以得到以下观察：

- `multihead` 的辅助监督分支在局部指标上有微小波动，但尚未形成稳定优势
- `consistency` 的多视角一致性版本目前低于对应基线
- 结构层面的“更接近目标”并不会自动转化为最终 item 排名提升
- SID 本身的 collision 和粗层级利用率不足，仍然限制了层级监督的有效上限

## 为什么当前版本尚未稳定超过基线

现阶段更合理的解释，是设计漂移与目标不一致共同作用的结果，而不是单一 bug。

主要原因包括：

1. 部分分支改变了原始 SFT 基座，不再与原版 mixed-task SFT 完全对齐
2. `multihead` 中三层辅助预测当前共用一个 hidden state，与真实自回归生成过程不完全一致
3. `prefix` 中 prefix-aware reward 可以改善粗粒度匹配，但不保证最终精确排序提升
4. `consistency` 中 full-vocab 一致性与表征对齐可能过强，削弱了多视角互补性
5. SID 层级结构本身并不完全干净，存在 collision 和 coarse level 利用不足的问题

## 仓库内容

本仓库保留了继续分析和推进实验所需的主要代码与文档，排除了大型训练产物和本地缓存。

包含内容：

- 训练与评测代码
- 数据预处理与格式转换脚本
- `rq/` 下的 SID 构建工具
- 分支实验脚本与结果说明文档

不包含内容：

- 模型权重和训练 checkpoint
- `output/`、`outputs/`、`results/`、`experiments/` 等生成目录
- RL 中间产物目录与重复本地数据副本

## 仓库结构

```text
.
├── prefix/
│   ├── README.md
│   ├── compare.md
│   ├── sft.py / rl.py / evaluate.py
│   ├── convert_dataset.py
│   ├── data/
│   ├── rq/
│   └── ...
├── multihead/
│   ├── README.md
│   ├── compare.md
│   ├── sft.py / rl.py / evaluate.py
│   ├── convert_dataset.py
│   ├── data/
│   ├── rq/
│   └── ...
└── consistency/
    ├── README.md
    ├── sft_mv.py
    ├── paired_mv_dataset.py
    ├── mv_consistency_trainer.py
    ├── data/
    ├── rq/
    └── ...
```

## 推荐阅读顺序

建议按以下顺序理解本仓库：

1. 根目录 `README.md`
2. `multihead/README.md`
3. `prefix/README.md`
4. `consistency/README.md`
5. `multihead/compare.md`
6. `prefix/compare.md`
7. `consistency/sft_mv.py`
8. `multihead/sft.py`
9. `prefix/rl.py`

## 运行说明

具体运行环境依赖本地数据、模型路径和显卡资源。典型流程如下：

```bash
conda create -n minionerec python=3.11 -y
conda activate minionerec

pip install -r prefix/requirements.txt
# 或安装 multihead/ 或 consistency/ 下的对应依赖

# 准备数据与 SID 索引
# 参考各分支中的 data/、rq/ 和 convert_dataset.py

bash multihead/sft.sh
bash multihead/rl.sh
bash multihead/evaluate.sh
```

如果复现 `prefix/` 或 `consistency/`，请改用对应目录下的脚本与入口文件。

## 后续工作建议

如果后续要把这些方向推进成更扎实的论文级结果，当前更合理的路线包括：

1. 恢复一个与原始 mixed-task SFT 严格可比的 base
2. 将层级辅助监督挂到真实 SID 自回归位置，而不是共享单一 hidden state
3. 让辅助损失和一致性损失渐进式 warmup
4. 统一使用离线 Top-K 指标评估 reward 设计
5. 优先提升 SID 质量，减少 collision，提高 coarse level 的有效利用率

## License

请以各分支目录中的 License 文件为准。
