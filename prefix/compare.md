# MiniOneRec-prefix 实验结果总表

这份文件专门整理当前仓库里“已经跑过”的结果，分成三类：

1. 已有离线测试 JSON，可直接比较 Top-K 指标的结果。
2. 训练已经跑完，但还没有统一离线测试 JSON 的结果。
3. 启动过、但中断或只留下部分 checkpoint 的结果。

如果你想看方法说明、代码差异、数据流和问答版说明，请看 `README_improvements.md`。

---

## 1. 结果来源

本文件使用的证据来自这些路径：

- `results/final_checkpoint/final_result_Industrial_and_Scientific.json`
- `experiments/runs/20260315_140310_pclpo_ablation/results/01_rule_test.json`
- `experiments/runs/20260315_140310_pclpo_ablation/results/02_hierarchical_test.json`
- `experiments/runs/20260315_140310_pclpo_ablation/csv/manual_test_summary.json`
- `rl_output/checkpoint-3300/trainer_state.json`
- `rl_output_02_hierarchical/checkpoint-6598/trainer_state.json`
- `rl_output_03_hierarchical_ranking/checkpoint-6598/trainer_state.json`
- `rl_output_04_pclpo_soft/checkpoint-6598/trainer_state.json`
- `rl_output_06_pclpo_margin_ranking/checkpoint-6598/trainer_state.json`
- `experiments/runs/20260315_140310_pclpo_ablation/01_rule/exp_config.json`
- `experiments/runs/20260315_140310_pclpo_ablation/02_hierarchical/exp_config.json`
- `experiments/runs/20260315_140310_pclpo_ablation/03_hierarchical_ranking/exp_config.json`
- `experiments/runs/20260315_140310_pclpo_ablation/errors/03_hierarchical_ranking.err.log`

说明：

- `manual_test_summary.json` 里的 `mean_hier_score` 定义是：

```text
0.1 * level0_acc + 0.2 * level1_acc + 0.3 * level2_acc + 1.0 * exact_match_acc
```

- 根目录 `results/final_checkpoint` 的样本数是 `5100`，而标准 ablation 测试集是 `4533`，所以它只能当参考，不能和 `01_rule/02_hierarchical` 做严格横比。

---

## 2. 先给结论

### 2.1 当前唯一完成了统一离线对比的结论

- `02_hierarchical` 没有超过 `01_rule`。
- 不是“只掉一个指标”，而是 `HR@3/5/10/20`、`NDCG@3/5/10/20`、`MRR`、`exact_match_acc` 全都下降。
- `level0_acc` 和 `level1_acc` 略有上升，但 `level2_acc` 和 exact 命中下降，说明它更像把预测拉向粗粒度前缀，而不是把正确 item 排到更前。

### 2.2 当前最值得继续补测的版本

- `pclpo_soft`

原因不是它已经证明最好，而是：

- 训练跑完了；
- 末端 `loss / KL / grad_norm` 比 `hierarchical_ranking` 和 `pclpo_margin_ranking` 更稳；
- `token_diversity` 也是这几种新奖励里最高的。

### 2.3 当前最应该先修再谈结论的地方

对 `hierarchical`、`hierarchical_ranking`、`pclpo_soft`、`pclpo_margin_ranking`，都应该先修：

- `reward_utils.safe_split_sid()` 的解析错误
- 非 prefix 级联的层级打分
- `sibling_map` 未被真正使用

否则训练 reward 和设计目标不是一回事。

---

## 3. 已有离线测试结果

### 3.1 可直接看的结果表

| 变体 | 结果文件 | 样本数 | Exact | HR@10 | HR@20 | NDCG@10 | NDCG@20 | MRR | 备注 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `baseline_final_checkpoint` | `results/final_checkpoint/final_result_Industrial_and_Scientific.json` | 5100 | 0.082353 | 0.158431 | 0.194706 | 0.115542 | 0.124703 | 0.106825 | 仅作参考，样本数与标准测试不一致 |
| `01_rule` | `experiments/runs/20260315_140310_pclpo_ablation/results/01_rule_test.json` | 4533 | 0.082727 | 0.157070 | 0.195897 | 0.115016 | 0.124766 | 0.106508 | 当前最可信的 RL 对照组 |
| `02_hierarchical` | `experiments/runs/20260315_140310_pclpo_ablation/results/02_hierarchical_test.json` | 4533 | 0.078976 | 0.149570 | 0.183102 | 0.109916 | 0.118473 | 0.102010 | 相比 `01_rule` 明确变差 |

### 3.2 Top-1 层级命中

| 变体 | Valid SID Rate | Level-0 | Level-1 | Level-2 | Mean Hier Score |
| --- | ---: | ---: | ---: | ---: | ---: |
| `baseline_final_checkpoint` | 1.000000 | 0.280588 | 0.146863 | 0.093333 | 0.167784 |
| `01_rule` | 1.000000 | 0.280168 | 0.143834 | 0.094860 | 0.167968 |
| `02_hierarchical` | 1.000000 | 0.282815 | 0.148908 | 0.088904 | 0.163711 |

这里的 `Mean Hier Score` 来自 `test.py`，计算方式是：

```text
0.1 * level0_acc + 0.2 * level1_acc + 0.3 * level2_acc + 1.0 * exact_match_acc
```

### 3.3 `02_hierarchical` 相对 `01_rule` 的变化量

| 指标 | Delta |
| --- | ---: |
| `HR@3` | -0.005736 |
| `HR@5` | -0.006618 |
| `HR@10` | -0.007501 |
| `HR@20` | -0.012795 |
| `NDCG@3` | -0.004425 |
| `NDCG@5` | -0.004767 |
| `NDCG@10` | -0.005100 |
| `NDCG@20` | -0.006293 |
| `MRR` | -0.004498 |
| `Exact Match` | -0.003750 |
| `Level-0 Acc` | +0.002647 |
| `Level-1 Acc` | +0.005074 |
| `Level-2 Acc` | -0.005956 |
| `Mean Hier Score` | -0.004258 |

### 3.4 这组结果应该怎么解释

最稳妥的解释是：

- `hierarchical` 的确把一部分预测往“相近 prefix”方向推了，所以 `level0_acc` 和 `level1_acc` 小幅上升。
- 但最终目标是推荐正确 item，而不是推荐到正确父类。
- 从 exact、MRR、HR、NDCG 全部下降来看，当前实现没有把 coarse-grained reward 转成真正的 exact ranking 提升。

这和代码风险是对得上的：

- reward 解析错了；
- 打分逻辑也不是严格 prefix；
- 所以提升粗粒度 prefix 命中并不奇怪，但无法转成最终 item 命中。

---

## 4. 训练完成但没有统一离线测试 JSON 的结果

这些目录都已经跑到 `global_step == max_steps`，说明训练流程完整结束了，但仓库里没有对应的 `final_result_*.json`。

### 4.1 总表

| 目录 | Reward Type | 证据类型 | Global Step | Last Loss | Last Reward | Last KL | Last Grad Norm | Token Diversity | 判断 |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `rl_output` | `ranking`，根据 `trainer_state` 中的 `rewards/rule_reward` 与 `rewards/ndcg_rule_reward` 反推 | 训练态 | 3300/3300 | 0.0226 | 0.0051 | 0.5664 | 1.2542 | 0.4688 | 老基线，训练稳定，但 reward 很低 |
| `rl_output_02_hierarchical` | `hierarchical` | 训练态 | 6598/6598 | 0.0320 | 0.2945 | 0.8008 | 2.5241 | 0.4609 | 训练完成，但只有日志，没有统一 test json |
| `rl_output_03_hierarchical_ranking` | `hierarchical_ranking` | 训练态 | 6598/6598 | 0.1097 | 0.4126 | 2.7500 | 4.2033 | 0.4375 | 更激进，KL 明显偏高 |
| `rl_output_04_pclpo_soft` | `pclpo_soft` | 训练态 | 6598/6598 | 0.0249 | 0.3508 | 0.6211 | 1.9370 | 0.5000 | 当前训练态最稳的新奖励版本 |
| `rl_output_06_pclpo_margin_ranking` | `pclpo_margin_ranking` | 训练态 | 6598/6598 | 0.1018 | 0.1870 | 2.5469 | 3.8484 | 0.4922 | KL 偏高，loss 偏大 |

### 4.2 如何解读这些训练态指标

`pclpo_soft`：

- `loss` 低
- `KL` 低
- `grad_norm` 低
- `token_diversity` 高

所以它是当前最值得补离线测试的版本。

`hierarchical_ranking` 和 `pclpo_margin_ranking`：

- `KL` 都明显偏高
- `loss` 也更高
- 说明策略比 `pclpo_soft` 更容易偏离 reference model

`rl_output` 这个老基线：

- 虽然 reward 数值低，但训练很稳
- 它更像“保守但稳定”的 ranking baseline

### 4.3 为什么这些还不能下最终结论

因为训练日志里的：

- `reward`
- `KL`
- `token_diversity`
- 在线小批次 `HR@K`

都不是完整测试集的最终 Top-K 指标。

没有统一离线测试 JSON，就不能说：

- 这个 reward 一定更好
- 或者这个 reward 一定更差

现在最多只能说“训练态更稳”或“训练态更激进”。

---

## 5. 已经启动但没有跑完的结果

### 5.1 `experiments/runs/20260315_140310_pclpo_ablation/03_hierarchical_ranking`

这是最关键的失败实验。

现状：

- 目录下有 `10` 个 checkpoint
- 数值上最新 checkpoint 到了 `checkpoint-2970`
- 但没有 `final_checkpoint`
- 也没有离线测试 JSON

错误位置来自多卡 gather：

- `minionerec_trainer.py`
- `rewards_per_func = gather(rewards_per_func)`

错误表现是：

- `Rank 0`: `Ranks 2 failed to pass monitoredBarrier`
- 其他 rank 提示：`successfully reached monitoredBarrier, but received errors while waiting for send/recv from rank 0`

因此，这个版本当前只能算“训练曾经跑到一半并多次重试过”，不能算完成结果。

### 5.2 更早的若干 ablation 尝试

| Run ID | 变体 | 当前状态 |
| --- | --- | --- |
| `20260315_063724_pclpo_ablation` | `01_rule` | 只留下 `3` 个 checkpoint，最新到 `checkpoint-1584` |
| `20260315_133613_pclpo_ablation` | `01_rule` | 没有 checkpoint，只有日志 |
| `20260315_134108_pclpo_ablation` | `01_rule` | 没有 checkpoint，只有日志 |

这些目录说明：

- 你的实验流程不是一次性稳定跑通的；
- 运行脚本和 resume 逻辑之间存在明显摩擦；
- 所以后面统一整理结果时，应该只把真正“完成训练”或“已有离线评测”的结果当主证据。

---

## 6. 每种变体具体做了什么

这一节是“变体定义 + 当前证据”的对照表。

### 6.1 `rule`

实现：

- `pred_sid == target_sid` 给 `1.0`
- 否则 `0.0`

优点：

- 目标直接，对齐 exact next-item recommendation

缺点：

- reward 非常稀疏

当前证据：

- 是最可信的 RL 对照组
- `Exact = 0.082727`
- `HR@20 = 0.195897`
- `MRR = 0.106508`

### 6.2 `ranking`

实现：

- `rule_reward + ndcg_rule_reward`

其中：

- `ndcg_rule_reward` 只在组内出现 exact hit 时，对未命中的 completion 给位置惩罚

当前证据：

- 根目录 `rl_output` 对应的是这条老基线，依据是 `trainer_state.json` 里同时记录了 `rewards/rule_reward` 和 `rewards/ndcg_rule_reward`
- 但没有保存 `exp_config.json`
- 也没有统一离线测试 JSON

因此这部分结论只能基于训练态。

### 6.3 `hierarchical`

实现：

- level-0 匹配给 `0.1`
- level-1 匹配给 `0.2`
- level-2 匹配给 `0.3`
- exact 匹配再给 `1.0`

当前证据：

- 有统一离线测试结果
- 也是所有新 reward 里唯一完成标准横比的版本

结果：

- 相比 `01_rule` 全面下降
- 只在 `level0_acc` 和 `level1_acc` 上略有上升

判断：

- 这更像“父类命中率提升”，不是“item 排序更好”

### 6.4 `hierarchical_ranking`

实现：

- `hierarchical_rule_reward + hierarchical_ndcg_reward`
- 后者把层级分数再乘上一个基于生成位置的 rank bonus

当前证据：

- 根目录 `rl_output_03_hierarchical_ranking` 有完整训练态结果
- `experiments/runs/20260315_140310_pclpo_ablation/03_hierarchical_ranking` 这条标准 ablation 跑失败了

结果：

- 训练态 `reward` 最高之一
- 但 `KL = 2.75`，偏高
- 标准 ablation 没跑完，不能下最终离线结论

### 6.5 `pclpo_soft`

实现：

- 先算 `hierarchical_match_score`
- 如果 `pred` 和 `target` 共享同一个 parent prefix，再额外加 `same_parent_bonus=0.15`

当前证据：

- 已训练完成
- 还没有统一离线测试 JSON

结果：

- `Last Loss = 0.0249`
- `Last KL = 0.6211`
- `Token Diversity = 0.5000`

判断：

- 是当前最值得优先补测的版本
- 但名字虽然叫 `pclpo_soft`，实现上更像“同 parent 软奖励”，不是真正 sibling-aware 的 PC-LPO

### 6.6 `pclpo_margin_ranking`

实现：

- `exact / same parent / wrong prefix` 三档区间式打分
- 再叠加 rank bonus

当前默认超参：

- `hier_reward_l0 = 0.05`
- `hier_reward_l1 = 0.10`
- `hier_reward_l2 = 0.20`
- `hier_reward_exact = 1.0`
- `same_parent_reward = 0.15`

当前证据：

- 已训练完成
- 还没有统一离线测试 JSON

结果：

- `Last Loss = 0.1018`
- `Last KL = 2.5469`
- `Token Diversity = 0.4922`

判断：

- 比 `pclpo_soft` 更激进
- 从训练态看风险更高

---

## 7. 为什么我不建议把当前训练日志当最终结论

因为现在最关键的对齐链条还没完全闭合：

1. `reward_utils.py` 的层级解析实现有问题。
2. 标准 ablation 只有 `rule` 和 `hierarchical` 做了统一离线测试。
3. 其余变体目前都只有训练态指标。

这意味着：

- 你可以说“`pclpo_soft` 训练更稳”
- 但不能说“`pclpo_soft` 最终推荐效果更好”

两者不是一回事。

---

## 8. 当前最可信的汇报口径

如果接下来要对导师、面试官或组会汇报，我建议你这样说：

1. 我基于 MiniOneRec 原版，在 RL 阶段实现了多种结构化 reward，包括 hierarchical、hierarchical_ranking、pclpo_soft、pclpo_margin_ranking。
2. 目前已经有标准离线对比的结果显示，`hierarchical` 没有超过 `rule`。
3. 其余新奖励大多已经完成训练，但还缺统一离线评测，因此暂时只能给出训练态分析。
4. 结合代码复盘，当前最可能的性能瓶颈在于层级 reward 的实现存在解析与打分不一致问题，下一步应该先修实现，再重跑统一测试。

这是当前最稳、也最技术上自洽的说法。
