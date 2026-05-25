# MiniOneRec Fork README

这个 README 不再重复上游仓库的宣传材料，而是专门说明你当前 3 个目录之间的关系、改动点、数据流和现有结果：

- `/root/MiniOneRec-main`
- `/root/MiniOneRec-multihead`
- `/root/MiniOneRec-consistency`

更详细的结果整理、运行痕迹和问题诊断在 [compare.md](./compare.md)。

## 1. 三个目录分别是什么

| 目录 | 定位 | 当前重点 |
| --- | --- | --- |
| `MiniOneRec-main` | 原始基线实现 | 原版多任务 SFT + 原版 RL |
| `MiniOneRec-multihead` | 你的多头 / 分层辅助监督分支 | 在单任务 SID SFT 上增加 3 个 level head，再接 RL |
| `MiniOneRec-consistency` | 你的多视角一致性分支 | 构造 SID 视角和 title 视角的 paired dataset，做一致性 SFT |

当前最重要的结论是：

- `multihead` 不是“原版 SFT + 多头”，而是“缩窄后的单任务 SFT + 多头”。
- `consistency` 不是“原版多任务 SFT + 一致性”，而是“paired two-view SFT + 一致性”。

所以这两个分支和 `main` 的差异，不只是加了新 loss，而是连 base 训练目标都变了。

## 2. 相对 `MiniOneRec-main` 的核心改动

### 2.1 `multihead`

关键文件：

- `data.py`
  - 新增 SID 拆分函数
  - 为 `SidSFTDataset` 生成 `target_sid_l0/l1/l2` 和 `level_pos`
- `sft.py`
  - 新增 `LevelAwareCausalLMWrapper`
  - 新增 `LevelAwareTrainer`
  - 在 AR loss 外增加 3 个 level CE loss
- `rl.py`
  - 增加 hierarchical reward
  - 增加配置检查、实验配置落盘、异常时应急保存
- `minionerec_trainer.py`
  - 给 `max_prompt_length` 加默认值
  - test 阶段单独构建 constrained logits processor

真正影响实验结论的关键点只有两个：

1. `sft.py` 不再混合 `SidItemFeatDataset` 和 `FusionSeqRecDataset`。
2. 3 个 level head 共用同一个 prompt 边界 hidden state。

### 2.2 `consistency`

关键新增文件：

- `paired_mv_dataset.py`
- `mv_consistency_trainer.py`
- `sft_mv.py`

这条分支的核心是：

1. 用同一条样本构造两个视角：
   - `SID history -> next SID`
   - `title history -> next SID`
2. 分别做前向传播
3. 用 CE + JS consistency + repr consistency 联合训练

需要注意的是：

- `consistency/sft.py` 基本保留了原版 `main/sft.py`
- 真正的新路线在 `sft_mv.py`

## 3. 整个模型的数据流

从原始数据到最终评测，完整链路如下。

### 3.1 数据预处理

入口：

- `data/amazon18_data_process.py`

作用：

- 清洗 review / metadata
- 做 k-core 过滤
- 按时间排序
- 切成 `train / valid / test`

### 3.2 文本编码

入口：

- `rq/text2emb/amazon_text2emb.py`

作用：

- 用冻结文本编码器把 item 文本变成连续 embedding

### 3.3 SID 构建

入口：

- `rq/rqvae.py`
- `rq/models/rqvae.py`
- `rq/models/rq.py`
- `rq/generate_indices.py`

作用：

- 用三层 residual quantization 把 embedding 量化成 SID
- 生成 `<a_*><b_*><c_*>` 格式的离散 token 序列

### 3.4 训练集转换

入口：

- `convert_dataset.py`

作用：

- 把 `item_id`、`item_sid`、历史 title、历史 SID 一起写进 SFT / RL 用的 CSV

### 3.5 原版 SFT

入口：

- `MiniOneRec-main/sft.py`

原版混合了 3 类任务：

- `SidSFTDataset`
- `SidItemFeatDataset`
- `FusionSeqRecDataset`

### 3.6 `multihead` SFT

入口：

- `MiniOneRec-multihead/sft.py`

当前流程是：

1. 只训练 `SidSFTDataset`
2. 从目标 SID 中拆出 3 层标签
3. 在一个 prompt 边界位置上预测 3 层 token
4. 优化 `AR loss + 3 个辅助分类 loss`

### 3.7 `consistency` SFT

入口：

- `MiniOneRec-consistency/sft_mv.py`

当前流程是：

1. 构造 SID 视角和 title 视角
2. 两个视角都预测同一个 next SID
3. 除了 CE loss，再对齐两个视角的 logits 与 hidden state

### 3.8 RL 与评测

入口：

- `rl.py`
- `minionerec_trainer.py`
- `evaluate.py`
- `LogitProcessor.py`

流程：

1. 用受限解码生成合法 SID 候选
2. 按 reward 计算 advantage
3. 做 GRPO 更新
4. 用 constrained decoding 做离线 Top-K 评测

## 4. 现有结果概览

当前本地正式、可复用的结果主要有 4 个：

| 变体 | 仓库 | 样本数 | Exact@1 | HR@10 | NDCG@10 | HR@20 | NDCG@20 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| baseline_sft | consistency | 4533 | 0.07456 | 0.14472 | 0.10479 | 0.18310 | 0.11440 |
| mv_sft | consistency | 4533 | 0.06684 | 0.13435 | 0.09539 | 0.16854 | 0.10394 |
| rl_base | multihead | 4533 | 0.07831 | 0.15442 | 0.11198 | 0.19347 | 0.12180 |
| rl_aux | multihead | 4533 | 0.07721 | 0.15773 | 0.11233 | 0.18840 | 0.12008 |

直接结论：

- `consistency` 的 `mv_sft` 明显低于 `baseline_sft`
- `multihead` 的 `rl_aux` 与 `rl_base` 基本打平，没有形成稳定优势

更详细的层级命中率、pairwise compare、训练日志统计和 run 目录整理，请看 [compare.md](./compare.md)。

## 5. 为什么现在没有超过原版

当前最可能的原因有 6 个。

1. 你改的不只是一个附加模块，而是把原版 SFT 的任务集合改掉了。
2. `multihead` 的 3 个辅助头共用同一个 hidden state，和真实自回归生成过程不一致。
3. 辅助 loss 的量级并不小，实际已经和 AR loss 同量级，优化方向会被明显重塑。
4. 当前最完整的 RL 结果使用的是 `rl.py` 默认配置，不是 `main/rl.sh` 的 `ranking` 配置。
5. `consistency` 的 full-vocab JS + anchor 表征对齐，可能把两个视角过度压成同一种表示。
6. SID 本身有碰撞，且 level-0 只用了 48 个 token，层级结构质量还不够好。

## 6. 更合理的下一步

如果你下一步是想继续把结果做上去，建议这样改：

1. 先恢复一个真正和 `main/sft.py` 对齐的 base。
2. 再在这个 base 上叠加 `multihead` 或 `consistency`。
3. `multihead` 要把 3 个 level head 挂到 3 个真实的 SID 生成位置上。
4. 给辅助损失做 warmup 或分阶段打开。
5. `consistency` 只在 SID 相关 token 上做一致性，不要直接对 full vocab 做 JS。
6. 所有 RL 实验参数都显式写在脚本里，不要依赖默认值。
7. 先修 SID 碰撞和 level-0 利用率。

## 7. 你现在最该看的文件

- [compare.md](./compare.md)
- `MiniOneRec-multihead/sft.py`
- `MiniOneRec-multihead/rl.py`
- `MiniOneRec-multihead/run_all_experiments.sh`
- `MiniOneRec-consistency/sft_mv.py`
- `MiniOneRec-consistency/paired_mv_dataset.py`
- `MiniOneRec-consistency/mv_consistency_trainer.py`

