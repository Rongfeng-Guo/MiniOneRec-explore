# MiniOneRec 三仓对比与结果整理

本文一次性对比 3 个目录：

- `/root/MiniOneRec-main`：原始基线实现
- `/root/MiniOneRec-multihead`：多头 / 分层辅助监督分支
- `/root/MiniOneRec-consistency`：多视角一致性分支

本文重点回答 4 个问题：

1. 相对 `MiniOneRec-main`，两个分支到底改了什么。
2. 本地已经跑完的结果有哪些，哪些能直接横向比较。
3. 为什么当前效果没有稳定超过原版。
4. 整个模型从数据到训练到评测的数据流怎么走。

## 1. 比较范围与证据来源

- 代码对比基于 `diff -qr` 和关键文件人工阅读。
- 结果整理只统计本地已经存在的 `results/*.json`、`summary/*.json`、`csv/*.csv`、`trainer_state.json`。
- `MiniOneRec-main` 目录下没有找到本地离线评测结果文件，所以“有没有超过原版”目前只能：
  - 用代码和配置对比来判断你是否真的在和原版做同一件事；
  - 用 `consistency` 里最接近原版 SFT 的 `baseline_sft_industrial_result.json` 作为本地代理参考；
  - 不能直接给出 `main` 在当前机器、当前 split 上的本地数值结论。

## 2. 结论先看

1. `multihead` 的核心改动不是“在原版 SFT 上轻量加一个多头正则”，而是先把原版 `main/sft.py` 的多任务 SFT 收缩成单一的 `SidSFTDataset`，再在其上加 3 个 SID level 分类头。
2. `consistency` 的核心改动也不是“在原版多任务 SFT 上再加一个一致性损失”，而是单独构造了双视角配对数据集，用 `SID history -> next SID` 和 `title history -> next SID` 两个分支做联合训练。
3. 从本地已经保存好的、可以直接比较的结果看：
   - `consistency` 的 `mv_sft` 明显低于它自己的 `baseline_sft`。
   - `multihead` 的 `rl_aux` 和 `rl_base` 基本打平，只有非常局部的小幅波动，没有形成稳定优势。
4. 造成“没有超过原版”的主要原因，不像是一个单点 bug，而更像是目标函数和实验配置一起漂移：
   - SFT 任务集合变了；
   - `multihead` 的 3 个 level head 共用同一个 prompt 边界 hidden state；
   - `run_all_experiments.sh` 真正跑出的 RL 结果实际上使用了 `rl.py` 的默认配置，而不是 `main/rl.sh` 的 `ranking` 配置。
5. SID 本身也有上限问题：
   - `Industrial_and_Scientific.index.json` 有 `3686` 个条目，但只有 `3670` 个唯一 SID；
   - 存在 `16` 个重复条目，对应 `15` 组 SID 碰撞；
   - 第 1 层 SID 实际只用了 `48` 个 token，远少于理论上的 `256`。

## 3. 仓库级改动总览

### 3.1 `MiniOneRec-main` 原版在做什么

原版训练主线是：

1. `data/amazon18_data_process.py`
   把 Amazon 原始 review / metadata 清洗、过滤、切分成序列数据。
2. `rq/text2emb/amazon_text2emb.py`
   把 item 的文本编码成连续 embedding。
3. `rq/rqvae.py` 与 `rq/generate_indices.py`
   把连续 embedding 量化成三层 SID。
4. `convert_dataset.py`
   把 `item_id` 和 `item_sid` 注入 SFT / RL 用的 CSV。
5. `sft.py`
   用 3 类任务混合训练：
   - `SidSFTDataset`
   - `SidItemFeatDataset`
   - `FusionSeqRecDataset`
6. `rl.py`
   用 3 类 prompt 混合做 GRPO：
   - `SidDataset`
   - `RLTitle2SidDataset`
   - `RLSeqTitle2SidDataset`
7. `evaluate.py` + `LogitProcessor.py`
   用受限解码做合法 SID Top-K 评测。

一句话概括原版：
它不是只学 `history SID -> next SID`，而是用“SID 序列、自然语言、item 文本特征”之间的多种互相映射，把语言模型和离散 item code 空间绑在一起。

### 3.2 `MiniOneRec-multihead` 相对 `main` 改了什么

代码级关键改动集中在 4 个文件：

| 文件 | 相对 `main` 的主要变化 | 直接影响 |
| --- | --- | --- |
| `data.py` | 新增 `split_sid_tokens_from_string`、`build_sid_token_to_level_id`，并让 `SidSFTDataset` 输出 `target_sid_l0/l1/l2` 与 `level_pos` | 为分层辅助监督准备标签 |
| `sft.py` | 新增 `LevelAwareCausalLMWrapper` 和 `LevelAwareTrainer`，在 AR loss 外再加 3 个 level CE loss | 把 SFT 从纯生成任务改成“生成 + 3 个分类头” |
| `rl.py` | 增加层级奖励、更多配置校验、实验配置落盘、异常时应急保存 | 改了 RL 训练逻辑和工程脚手架 |
| `minionerec_trainer.py` | 为 `max_prompt_length` 提供默认值，并给 test 单独构建 constrained logits processor | 修正部分生成 / 评测行为 |

工程上还新增了不少脚本：

- `run_all_experiments.sh`
- `run_mutihead.sh`
- `run_rl_base.sh`
- `run_rl_aux.sh`
- `run_rl_aux_hier.sh`
- `run_rl_aux_hier_stable.sh`
- `collect_results.py`
- `extract_log_metrics.py`

但是最关键的不是“多了多少脚本”，而是 `sft.py` 的训练数据已经变了：

- 原版 `main/sft.py`：`SidSFTDataset + SidItemFeatDataset + FusionSeqRecDataset`
- `multihead/sft.py`：只使用 `SidSFTDataset`

这意味着：

- `multihead` 里的 `sft_base` 不是原版 `main` 的 base；
- `multihead` 里的 `sft_aux` 也不是在原版 full SFT 上加 head，而是在一个更窄的单任务 base 上加 head。

### 3.3 `MiniOneRec-consistency` 相对 `main` 改了什么

`consistency` 的真正新东西主要在 3 个新增文件：

| 文件 | 作用 |
| --- | --- |
| `paired_mv_dataset.py` | 把同一条样本构造成两个视角：`SID history -> next SID` 和 `Title history -> next SID` |
| `mv_consistency_trainer.py` | 在两次前向传播的 CE loss 外，再加分布一致性和表征一致性损失 |
| `sft_mv.py` | 多视角一致性训练入口 |

其他变化里，最重要的是：

- `consistency/sft.py` 与 `main/sft.py` 只差很少，基本仍是原版 SFT。
- `consistency/rl.py` 基本延续 `main/rl.py`，没有专门做新的 RL 变体。
- 真正的新训练路线是 `sft_mv.py`，而不是替换掉整个 RL。

所以这个分支的实质是：
它新增了一条“多视角一致性 SFT”路线，而不是系统性重写整个 MiniOneRec。

## 4. 数据与 SID 统计

这里统一使用本地 `Industrial_and_Scientific_5_2016-10-2018-11` 数据。

### 4.1 split 统计

| 项目 | 数值 |
| --- | ---: |
| train 行数 | 36,259 |
| valid 行数 | 4,532 |
| test 行数 | 4,533 |
| 总样本数 | 45,324 |
| 用户数 | 7,694 |
| split 中唯一 `item_id` 数 | 3,683 |
| split 中唯一 `item_sid` 数 | 3,667 |
| `info.txt` 条目数 | 3,686 |
| 由历史重建得到的总交互数 | 225,528 |
| 平均历史长度 | 3.9759 |
| 最大历史长度 | 10 |

### 4.2 SID 统计

| 项目 | 数值 |
| --- | ---: |
| `index.json` 条目数 | 3,686 |
| 唯一 SID 数 | 3,670 |
| 重复 SID 条目数 | 16 |
| SID 碰撞组数 | 15 |
| level-0 唯一 token 数 | 48 |
| level-1 唯一 token 数 | 256 |
| level-2 唯一 token 数 | 256 |

这说明两个问题：

1. 你的 level-0 非常稀疏，只有 `48/256` 被用到，粗粒度分类头的信息量有限。
2. SID 仍有碰撞，训练目标并不是完美的一一对应映射。

## 5. 整个模型的数据流

这一节按真实代码路径把整个系统串起来。

### 5.1 原始交互预处理

入口：`data/amazon18_data_process.py`

主要流程：

1. 读取 metadata 和 review。
2. 清洗 title / description。
3. 做 user / item 的 k-core 过滤。
4. 按时间排序每个用户的交互。
5. 生成历史序列和下一个物品。
6. 产出 `train/valid/test`、`item.json`、`info.txt` 等中间文件。

### 5.2 item 文本编码

入口：`rq/text2emb/amazon_text2emb.py`

主要流程：

1. 读取 item 的 `title + description`。
2. 送入冻结文本编码器。
3. 对 hidden state 做 pooling。
4. 保存为 `.npy` embedding。

输出是每个 item 的连续语义向量。

### 5.3 embedding 到 SID

入口：

- `rq/rqvae.py`
- `rq/models/rqvae.py`
- `rq/models/rq.py`
- `rq/generate_indices.py`

主要流程：

1. 编码器把 embedding 压到较小维度。
2. Residual vector quantizer 逐层量化 residual。
3. 每层选一个 code index。
4. decoder 重构 embedding。
5. 最终把 3 层 index 转成 3 个 SID token，例如：
   - `<a_i><b_j><c_k>`
6. `generate_indices.py` 再做碰撞检查和重分配。

### 5.4 SID 注入训练数据

入口：`convert_dataset.py`

转换后 CSV 里的关键字段是：

- `history_item_sid`
- `item_sid`
- `history_item_title`
- `item_title`
- `history_item_id`
- `item_id`

这一步之后，同一条样本同时保留了 SID 序列和文本侧信息，供 SFT / RL 共用。

### 5.5 原版 SFT 数据流

入口：`main/sft.py`

原版是 3 个数据集混合训练：

1. `SidSFTDataset`
   任务是 `history SID -> next SID`
2. `SidItemFeatDataset`
   任务是 title / SID 间的双向映射
3. `FusionSeqRecDataset`
   任务是序列语义与 item 文本特征对齐

这三者一起，才是原版 MiniOneRec 的 SFT。

### 5.6 `multihead` 的 SFT 数据流

入口：`multihead/sft.py`

训练流程变成：

1. 只保留 `SidSFTDataset`
2. 读取目标 SID，拆成 3 层 token 标签
3. 取 `level_pos = input_prompt_len - 1` 的 hidden state
4. 用这个单一 hidden state 同时预测：
   - level-0
   - level-1
   - level-2
5. 总损失：
   - `AR loss`
   - `+ 0.1 * loss_l0`
   - `+ 0.1 * loss_l1`
   - `+ 0.1 * loss_l2`

注意这里有一个非常关键的实现细节：
3 个 level head 共用的是同一个 prompt 边界位置 hidden state，而不是分别对 `<a_>`、`<b_>`、`<c_>` 的生成位置建模。

### 5.7 `consistency` 的 SFT 数据流

入口：`consistency/sft_mv.py`

训练流程是：

1. `PairedSidTitleSFTDataset` 对同一条样本构造两个 branch：
   - SID 视角：`SID history -> next SID`
   - Title 视角：`title history -> next SID`
2. 模型对两个 branch 分别做前向传播。
3. `MultiViewConsistencyTrainer` 计算：
   - 两个 branch 的平均 CE loss
   - target SID 位置 logits 的 JS 一致性
   - prompt 边界 anchor hidden 的 cosine 一致性
4. 总损失：
   - `ce_loss + lambda_dist * dist_loss + lambda_repr * repr_loss`

默认权重是：

- `lambda_dist = 0.2`
- `lambda_repr = 0.05`
- `temperature = 1.0`

### 5.8 RL 与评测数据流

原版和两个分支都沿用了以下思路：

1. RL 输入由多种 prompt 组成。
2. 一次 prompt 生成多条候选。
3. 根据规则 reward / ranking reward / 其他 reward 算分。
4. 组内标准化 advantage。
5. 加上 KL 正则做 GRPO 更新。
6. 评测时用 `LogitProcessor.py` 做受限解码，只允许走合法 SID 前缀。

从结果文件看，几个正式结果的 `valid_sid_rate` 都是 `1.0`，说明受限解码是生效的。

## 6. 各变体怎么实现

### 6.1 原版 `main`

- `main/sft.py`：原始三任务混合 SFT
- `main/rl.py`：原始三任务混合 RL
- `main/rl.sh`：给出了论文式 RL 脚本入口

### 6.2 `multihead/sft_base`

实现方式：

- 使用 `SidSFTDataset`
- 不再混合 `SidItemFeatDataset` 和 `FusionSeqRecDataset`
- 训练目标只有 `history SID -> next SID`

这不是原版 base，而是一个缩窄后的单任务 base。

### 6.3 `multihead/sft_aux`

实现方式：

- 训练数据仍只有 `SidSFTDataset`
- 在 CausalLM 外包一层 `LevelAwareCausalLMWrapper`
- 从 `index.json` 中拆出 3 层 SID 标签
- 用 3 个线性头预测 3 层 token

关键问题：

- 3 个 level 都由同一个 `level_pos` hidden state 预测；
- 没有利用 `<a_>` 生成后再预测 `<b_>`、`<c_>` 的自回归条件结构。

### 6.4 `multihead/rl_base`

实现方式：

- 从 `sft_base` checkpoint 出发做 RL
- RL 数据仍然沿用 3 类 prompt 混合
- 完整可比结果来自：
  - `experiments/runs/20260312_101700/rl_base/exp_config.json`
  - `results/20260312_101700_rl_base_test.json`

### 6.5 `multihead/rl_aux`

实现方式：

- 从 `sft_aux` checkpoint 出发做 RL
- RL 阶段本身不再使用 level head
- 也就是说，多头监督只在 SFT 阶段当 regularizer 使用

完整可比结果来自：

- `experiments/runs/20260312_101700/rl_aux/exp_config.json`
- `results/20260312_101700_rl_aux_test.json`

### 6.6 `multihead` 里实现了但没有正式对比结果的变体

从脚本和 smoke config 看，还实现了这些 reward 形式：

- `hierarchical`
- `hierarchical_ranking`
- `hierarchical_ranking_only`

对应脚本包括：

- `run_rl_aux_hier.sh`
- `run_rl_aux_hier_stable.sh`

但当前仓库里没有与 `20260312_101700_rl_base_test.json` / `rl_aux_test.json` 同等级的正式测试结果文件，所以这些变体目前只能算“实现了”，不能算“已完成主实验”。

### 6.7 `consistency/baseline_sft`

这是当前本地最接近 `main/sft.py` 的 SFT 结果代理：

- `consistency/sft.py` 与 `main/sft.py` 的差异很小
- 本地结果文件是 `results/baseline_sft_industrial_result.json`

所以如果你要在没有 `main` 本地 result json 的前提下找“最像原版 SFT 的本地参考”，这个文件最有参考价值。

### 6.8 `consistency/mv_sft`

实现方式：

1. 同一条样本构造 SID 视角和 title 视角。
2. 两个视角分别生成同一个 next SID。
3. 用 CE loss 保证各自能做对。
4. 用分布一致性与表征一致性把两个视角拉近。

它的设计目标是：

- 让不同输入形式学到相同的下一物品语义；
- 让 SID 空间在不同视角下更稳定。

### 6.9 `consistency` 中已训练但没有统一 test json 的额外变体

除了正式结果文件外，还发现了这些训练输出目录：

| 目录 | 状态 | 当前可读到的信息 |
| --- | --- | --- |
| `output/mv_sft_industrial_2gpu` | 训练过 | 2 epoch，best eval loss `1.6189`，无离线 Top-K 结果文件 |
| `output/mv_sft_industrial_gpu0_noeval` | 训练过 | 2 epoch，best eval loss `1.6130`，有 `final_checkpoint`，无离线 Top-K 结果文件 |
| `output/mv_sft_industrial_weak_both` | 训练过 | 10 epoch，无 eval，最终 train loss 很低，但没有统一 test 结果文件 |

这些目录能证明你做过额外一致性实验，但因为缺少统一的离线测试导出，所以不纳入严格结果表。

## 7. 已完成且可直接引用的结果

### 7.1 严格可比的正式结果

下面 4 个结果都基于 `4533` 个 test 样本，可以直接横向比较：

| 变体 | 仓库 | 样本数 | Exact@1 | HR@10 | NDCG@10 | HR@20 | NDCG@20 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| baseline_sft | consistency | 4533 | 0.07456 | 0.14472 | 0.10479 | 0.18310 | 0.11440 |
| mv_sft | consistency | 4533 | 0.06684 | 0.13435 | 0.09539 | 0.16854 | 0.10394 |
| rl_base | multihead | 4533 | 0.07831 | 0.15442 | 0.11198 | 0.19347 | 0.12180 |
| rl_aux | multihead | 4533 | 0.07721 | 0.15773 | 0.11233 | 0.18840 | 0.12008 |

### 7.2 分层命中情况

这里给出更直观、可复算的层级指标：

- `SimpleHierAvg = (level0_acc + level1_acc + level2_acc) / 3`

| 变体 | level-0 acc | level-1 acc | level-2 acc | SimpleHierAvg |
| --- | ---: | ---: | ---: | ---: |
| baseline_sft | 0.27377 | 0.13722 | 0.08449 | 0.16516 |
| mv_sft | 0.26517 | 0.12949 | 0.08206 | 0.15891 |
| rl_base | 0.27201 | 0.14560 | 0.08537 | 0.16766 |
| rl_aux | 0.26539 | 0.14273 | 0.09243 | 0.16685 |

### 7.3 `consistency` 自己的对比

`mv_sft` 相对 `baseline_sft` 的变化：

- Exact@1：`-0.00772`
- HR@10：`-0.01037`
- NDCG@10：`-0.00940`
- HR@20：`-0.01456`
- NDCG@20：`-0.01046`
- level-0 acc：`-0.00860`
- level-1 acc：`-0.00772`
- level-2 acc：`-0.00243`

这个分支从当前证据看，是明确退化的。

### 7.4 `multihead` 自己的对比

`rl_aux` 相对 `rl_base` 的变化：

- Exact@1：`-0.00110`
- HR@10：`+0.00331`
- NDCG@10：`+0.00034`
- HR@20：`-0.00507`
- NDCG@20：`-0.00172`
- level-0 acc：`-0.00662`
- level-1 acc：`-0.00287`
- level-2 acc：`+0.00706`

这说明它更像是把错误形态做了转移，而不是整体提升。

### 7.5 `multihead` 两个正式结果的逐样本关系

来自 `experiments/runs/20260312_101700/summary/test_metric_summary.json`：

| 指标 | 数值 |
| --- | ---: |
| both_correct | 308 |
| base_only_correct | 47 |
| aux_only_correct | 42 |
| both_wrong | 4,136 |
| same_prediction | 2,187 |
| same_prediction_rate | 0.48246 |

这组数字很重要：

- `base_only_correct > aux_only_correct`
- 两者有接近一半样本给出完全相同的预测

所以 `aux` 不是形成了稳定增益，而是在少量样本上与 `base` 互有胜负。

### 7.6 `multihead` 训练日志摘要

来自 `experiments/runs/20260312_101700/csv/train_metric_summary.csv`：

| 变体 | mean loss | mean reward | mean KL | mean train HR@10 | mean train NDCG@10 | max grad norm |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| rl_base | 0.00695 | 0.01959 | 0.17371 | 0.30076 | 0.22751 | 43.93 |
| rl_aux | 0.00751 | 0.01958 | 0.18764 | 0.29947 | 0.22392 | 16.96 |

可以看到：

- `rl_aux` 的平均 reward 并没有更高；
- `rl_aux` 的 mean KL 略高；
- `rl_base` 的最大梯度范数更大，但最终离线指标并不更差。

也就是说，`aux` 并没有在训练统计上展示出清晰的“更好优化方向”。

### 7.7 归档参考结果

这两个文件的 SHA256 完全相同：

- `MiniOneRec-multihead/results/final_checkpoint/final_result_Industrial_and_Scientific.json`
- `MiniOneRec-consistency/results/final_checkpoint/final_result_Industrial_and_Scientific.json`

它们的样本数都是 `5100`，与当前本地 test split 的 `4533` 不一致，所以只能当归档参考，不能和上面 4 个正式结果混排。

| 变体 | 仓库 | 样本数 | Exact@1 | HR@10 | NDCG@10 | HR@20 | NDCG@20 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| final_checkpoint | multihead / consistency | 5100 | 0.08235 | 0.15843 | 0.11554 | 0.19471 | 0.12470 |

## 8. 已跑过的目录与状态整理

### 8.1 `multihead` 目录

已确认的关键 run 如下：

- `20260312_101700`
  - 有 `rl_base` 与 `rl_aux` 的完整输出
  - 有 `exp_config.json`
  - 有 `parsed_log_metrics.json`
  - 有正式 test 汇总文件
  - 这是当前最完整、最可比的一组结果
- `20260311_154716`
  - 有 `sft_base/final_checkpoint`
  - 有 `sft_aux/final_checkpoint`
  - 有 `rl_base` checkpoint 和 resume 痕迹
  - 但没有统一 test 结果导出
- `20260311_150541`
  - 有 `sft_base/final_checkpoint`
  - 有 `sft_aux/final_checkpoint`
  - 但没有统一 test 结果导出
- `20260311_135755`
  - 有 `sft_base/final_checkpoint`
  - 有 `sft_aux/final_checkpoint`
  - 但没有统一 test 结果导出
- `20260311_121230`
  - 有 `sft_base/final_checkpoint`
  - 有 `sft_aux/final_checkpoint`
  - 没有正式 test 结果导出
- `20260310_092321`
  - 有早期 `sft_base`、`sft_aux` 输出
  - 有 smoke 级 `hierarchical_ranking` RL config
  - 没有正式 test 结果导出
- 其他目录
  - 大多是 smoke、试跑或中断痕迹
  - 不适合纳入正式对比表

### 8.2 `consistency` 目录

已确认的关键结果来源如下：

- `results/baseline_sft_industrial_result.json`
  - 正式离线结果
- `results/mv_sft_industrial_result.json`
  - 正式离线结果
- `output/checkpoint-78` 与 `output/final_checkpoint`
  - baseline SFT 训练产物
- `output/mv_sft_industrial_2gpu`
  - 多视角训练产物
- `output/mv_sft_industrial_gpu0_noeval`
  - 多视角训练产物
- `output/mv_sft_industrial_weak_both`
  - 额外一致性训练产物

因此，`consistency` 分支不缺训练痕迹，缺的是把所有训练产物都统一导出成同一格式的 test 结果文件。

## 9. 为什么当前效果不理想

### 9.1 你没有在“原版 SFT”上加模块，而是在改 SFT 的任务分布

这是最大的问题。

原版 `main/sft.py` 是 3 个任务混训，而你两个分支的核心新路线都把训练目标收窄了：

- `multihead/sft.py`：只剩 `SidSFTDataset`
- `consistency/sft_mv.py`：只剩 paired two-view SID 预测

结果就是：

- 你看到的差距不只是“aux / consistency 有没有用”；
- 还混入了“base 本身是不是已经比原版更弱”的影响。

### 9.2 `multihead` 的 3 个 head 共用同一个 hidden state

`multihead/sft.py` 的逻辑是：

- `level_pos = input_prompt_len - 1`
- 从这一个位置取 hidden state
- 同时预测 `l0`、`l1`、`l2`

这会导致两个问题：

1. 它没有利用 SID 三个 token 本身的自回归依赖。
2. 同一个 hidden state 要同时承担“粗粒度分类”和“细粒度分类”，目标之间可能互相拉扯。

这和真正的生成过程不一致。

更合理的实现通常应该是：

- prompt 边界位置预测 `<a_*>`
- 看到 `<a_*>` 后再预测 `<b_*>`
- 看到 `<a_*> <b_*>` 后再预测 `<c_*>`

### 9.3 `multihead` 的辅助损失并不“小”

从 `output/checkpoint-36/trainer_state.json` 的日志统计看，`aux` SFT 阶段平均：

- `loss_ar ≈ 1.40`
- `loss_l0 ≈ 3.69`
- `loss_l1 ≈ 5.51`
- `loss_l2 ≈ 5.57`

即使每个辅助头只乘 `0.1`，辅助项总贡献大约也是：

- `0.1 * (3.69 + 5.51 + 5.57) ≈ 1.48`

这个量级已经和 `AR loss ≈ 1.40` 差不多了。

所以它不是一个很轻的 regularizer，而是在显著重塑优化方向。

### 9.4 正式 RL 结果的实际配置，与 `main/rl.sh` 并不对齐

这个点非常关键。

`main/rl.sh` 里写的是：

- `reward_type = ranking`
- `train_batch_size = 64`
- `eval_batch_size = 128`
- `learning_rate = 1e-5`
- `beta = 1e-3`
- `sync_ref_model = True`

而你当前最完整的 `20260312_101700` 结果，实际由 `run_all.log` 和 `exp_config.json` 还原出来的是：

- `reward_type = rule`
- `train_batch_size = 32`
- `eval_batch_size = 64`
- `learning_rate = 1e-6`
- `beta = 0.04`
- `sync_ref_model = False`

更具体地说：

- `run_all_experiments.sh` 只显式传了 `--train_batch_size`、`--eval_batch_size`、`--num_generations`、`--beam_search` 等参数；
- 没有显式传 `reward_type`、`learning_rate`、`beta`、`sync_ref_model`；
- 所以最终结果使用的是 `rl.py` 默认值。

这意味着你当前的正式结果，并不是“对齐原版 RL 配置后，multihead 有没有带来收益”的答案。

### 9.5 `consistency` 的一致性约束可能过度约束了无关维度

`mv_consistency_trainer.py` 里的一致性有两类：

1. 分布一致性
   - 在 SID 三个目标位置上取 full-vocab logits 做 JS
2. 表征一致性
   - 对两个 branch 的 prompt 边界 hidden state 做 cosine 对齐

问题在于：

- full-vocab JS 会把大量与 SID 预测无关的词表概率也拉进来；
- prompt 边界 hidden state 对齐，会把“SID 历史视角”和“title 历史视角”的表示强行压到一起；
- 但这两个视角本来就承载不同的信息。

这类约束如果不够轻，很容易把两个 branch 都拉向一个折中表示，最后谁都不够好。

### 9.6 SID 结构本身有碰撞和粗层利用率不足

你当前的 SID 空间还有两个明显问题：

- 有碰撞，意味着不同 item 会共享同一个 SID；
- level-0 只有 48 个 token，粗层过窄。

这会造成：

- coarse-level 监督容易学、但区分度不够；
- fine-level 目标又因为碰撞和长尾而更难学；
- 最后出现“level-2 稍微变好，但 exact match 不一定变好”的情况。

### 9.7 工程脚本还有不一致问题

有两个明显的工程问题：

1. `run_all_experiments.sh` 在 Stage 5 调 `collect_results.py` 时传的是 `--root_dir`，但脚本实际要求 `--run_root`，所以日志里直接报错退出。
2. `collect_results.py` 和 `extract_log_metrics.py` 当前内容几乎重复，接口也没有统一好。

这类问题不直接影响模型效果，但会影响你对实验的可追踪性和可复现实验矩阵。

## 10. 更靠谱的后续修改建议

如果你的目标是“判断 multihead / consistency 能不能超过原版”，建议按下面顺序重做：

1. 先做一个真正和 `main/sft.py` 对齐的 base。
   - 保留 `SidSFTDataset + SidItemFeatDataset + FusionSeqRecDataset`
   - 不要先把 base 收窄
2. 再在这个 true base 上叠加 `multihead` 或 `consistency`。
   - 这样比较才有意义
3. 对 `multihead`，把 3 个 head 挂到 3 个真实的 AR 位置上。
   - 不要让同一个 hidden state 同时预测 3 层
4. 给辅助损失加 warmup 或分阶段权重。
   - 例如只先开 `l0`
   - 后面再逐步打开 `l1`、`l2`
5. 对 `consistency`，把一致性约束限制在 SID 相关 token 上。
   - 不要直接做 full-vocab JS
   - 可以只对 SID token 子词表做归一化后再比较
6. 统一 RL 脚本和正式 run 的参数。
   - 所有关键参数都显式传入
   - 不依赖 `rl.py` 默认值
7. 先修 SID 空间。
   - 降低碰撞
   - 提高 level-0 使用率

## 11. 本文用到的关键文件

代码证据：

- `MiniOneRec-main/sft.py`
- `MiniOneRec-main/rl.py`
- `MiniOneRec-main/rl.sh`
- `MiniOneRec-multihead/data.py`
- `MiniOneRec-multihead/sft.py`
- `MiniOneRec-multihead/rl.py`
- `MiniOneRec-multihead/minionerec_trainer.py`
- `MiniOneRec-multihead/run_all_experiments.sh`
- `MiniOneRec-consistency/sft.py`
- `MiniOneRec-consistency/sft_mv.py`
- `MiniOneRec-consistency/paired_mv_dataset.py`
- `MiniOneRec-consistency/mv_consistency_trainer.py`

结果证据：

- `MiniOneRec-multihead/results/20260312_101700_rl_base_test.json`
- `MiniOneRec-multihead/results/20260312_101700_rl_aux_test.json`
- `MiniOneRec-multihead/experiments/runs/20260312_101700/csv/test_metric_summary.csv`
- `MiniOneRec-multihead/experiments/runs/20260312_101700/csv/train_metric_summary.csv`
- `MiniOneRec-multihead/experiments/runs/20260312_101700/summary/test_metric_summary.json`
- `MiniOneRec-consistency/results/baseline_sft_industrial_result.json`
- `MiniOneRec-consistency/results/mv_sft_industrial_result.json`
- `MiniOneRec-consistency/output/mv_sft_industrial_2gpu/checkpoint-568/trainer_state.json`
- `MiniOneRec-consistency/output/mv_sft_industrial_gpu0_noeval/checkpoint-568/trainer_state.json`
- `MiniOneRec-consistency/output/mv_sft_industrial_weak_both/checkpoint-2840/trainer_state.json`

