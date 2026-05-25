# MiniOneRec-prefix 改进说明、数据流拆解与复盘

这份文档面向当前仓库 `/root/MiniOneRec-prefix`，重点回答四件事：

1. 相对原版 `/root/MiniOneRec-main`，你实际改了哪些地方。
2. 整个模型从 item 元数据到最终推荐结果的数据流是怎么走的。
3. 现在效果没有超过原版，最可能是哪里实现得不够好。
4. 如果要汇报这份工作，关于数据、SID、RQ-VAE、SFT、GRPO 应该怎么讲。

实验结果的完整盘点放在 `compare.md`。本文件更偏向“方案说明 + 代码复盘 + 面试问答”。

---

## 1. 一页结论

### 1.1 这次修改真正动到的核心

和 `/root/MiniOneRec-main` 相比，这个仓库的主要变化几乎都集中在 RL 奖励设计和实验组织上：

- 新增了 `reward_utils.py`，实现了 `hierarchical`、`pclpo_soft`、`pclpo_margin` 相关逻辑。
- `rl.py` 从原来的 `rule / ranking / semantic / sasrec` 扩展到了多种层级 reward 变体。
- 新增了 `run.sh`、`collect_results.py`、`extract_log_metrics.py`，方便做 ablation 和日志整理。
- `minionerec_trainer.py` 对测试阶段 constrained decoding 做了调整，让测试 beam 数和训练阶段解耦。

SFT 本体只做了很小的工程性改动，RQ-VAE 这条 SID 构建主链路也没有被大改。也就是说，这次“改进”真正的创新点不在 SFT 和 SID 生成，而在 RL 奖励函数。

### 1.2 目前最可靠的实验结论

基于仓库里已经落地的离线评测结果，当前最可靠的结论是：

- `01_rule` 是当前最可信的 RL 对照组。
- `02_hierarchical` 相比 `01_rule` 没有提升，且 `HR@K / NDCG@K / MRR / exact match` 全面下降。
- `pclpo_soft`、`pclpo_margin_ranking`、`hierarchical_ranking` 虽然有训练完成的 checkpoint，但没有统一离线测试 JSON，因此不能把训练日志当成最终效果。

### 1.3 为什么我认为“效果不理想”更像实现问题，而不是思路完全错

最关键的原因有三个，而且都是代码级问题，不是抽象想法的问题：

1. `reward_utils.py` 里的 `safe_split_sid()` 会把 SID 按 `_` 切开，但你的 SID 实际格式是 `<a_223><b_80><c_165>`。
2. `hierarchical_match_score()` 不是严格按 prefix 层级打分，而是分别比较第 0/1/2 层 token，父层不相同也可能拿到子层分。
3. `pclpo_*` 虽然构建了 `sibling_map`，但打分时并没有真正使用 sibling 关系，实际上只是“共享前缀就加分”。

这三个问题叠加后，训练期 reward 表示的“层级相似”与设计意图不一致，而离线评测脚本 `test.py` 却是用正则正确解析 `<a_x><b_y><c_z>`。换句话说：

- 训练时优化的层级语义是错位的。
- 测试时评估的层级语义是正确的。

这会直接导致“训练觉得自己在优化层级关系，测试却看不到收益”。

### 1.4 我建议的修复优先级

如果后面还要继续做，我建议按这个顺序推进：

1. 先修 `reward_utils.py` 的 SID 解析和 prefix 级联打分逻辑。
2. 再统一离线评测 `rule / hierarchical / hierarchical_ranking / pclpo_soft / pclpo_margin_ranking`。
3. 最后再讨论 reward 设计本身是否真的有效。

如果不先修实现，后面继续做 reward ablation，结论会一直掺杂实现噪声。

---

## 2. 相对原版改了什么

下面只列“逻辑上有影响”的改动，不把 checkpoint、日志和结果文件当成方法改动。

| 文件 | 原版情况 | 现在的改动 | 影响判断 |
| --- | --- | --- | --- |
| `reward_utils.py` | 原版没有 | 新增 SID 归一化、层级匹配、PC-LPO 打分、NDCG penalty | 这是新方法的核心实现 |
| `rl.py` | 只有 `rule / ranking / semantic / sasrec` | 扩展为 `hierarchical`、`hierarchical_ranking`、`pclpo_soft`、`pclpo_margin_ranking` 等 | 这是你真正的算法改动主战场 |
| `minionerec_trainer.py` | 训练和测试 constrained decoding 共用一套 beam 设置 | 测试阶段单独创建 `test_ccc`，支持 `test_beam` 与训练 beam 分离 | 这是合理的工程修正 |
| `run.sh` | 原版没有 | 新增一套 ablation 运行、日志、结果整理脚本 | 方便做系统实验，但 resume 逻辑未打通 |
| `rl.sh` | 模板式启动脚本 | 填了多种 reward 变体的真实训练命令 | 工程上更完整 |
| `collect_results.py` / `extract_log_metrics.py` | 原版没有 | 新增日志解析与结果整理 | 对复盘很有帮助 |
| `sft.py` | `group_by_length=group_by_length`，`report_to=None` | 注释掉 `group_by_length`，改成 `report_to=\"none\"` | 对性能影响很小 |
| `sft.sh` | 占位配置 | 改成可直接运行的实际路径和输出目录 | 主要是工程可用性提升 |
| `evaluate.sh` | 占位模型路径，默认 GPU `0-7` | 固定到 `./rl_output/final_checkpoint`，GPU 改成 `4-9` | 可以跑，但引入了一个文件检查 bug |

### 2.1 一句话概括这次工作的定位

原版更像是：

```text
SFT + rule/ranking RL
```

你这版更像是：

```text
SFT + 结构化 reward 探索（hierarchical / parent-child / margin）
```

也就是说，这次工作的核心不是重写 MiniOneRec，而是在 RL reward 上做结构增强。

---

## 3. Amazon Industrial_and_Scientific 数据统计

以下数字都来自当前仓库的现成文件：

- `data/Amazon/train/Industrial_and_Scientific_5_2016-10-2018-11.csv`
- `data/Amazon/valid/Industrial_and_Scientific_5_2016-10-2018-11.csv`
- `data/Amazon/test/Industrial_and_Scientific_5_2016-10-2018-11.csv`
- `data/Amazon/index/Industrial_and_Scientific.index.json`
- `data/Amazon/index/Industrial_and_Scientific.item.json`

### 3.1 全局统计

| 指标 | 数值 |
| --- | ---: |
| 原始 item 数 | 3686 |
| 全部用户数 | 7694 |
| train 样本数 | 36259 |
| valid 样本数 | 4532 |
| test 样本数 | 4533 |
| train+valid+test 总样本数 | 45324 |
| 总交互事件数（历史 + target） | 225528 |
| 全部样本中出现过的唯一 SID 数 | 3670 |

这里要区分两个概念：

- 如果问题问的是“用了多少个 item”，答案是 `3686`。
- 如果问题问的是“模型最终看到多少个唯一 SID”，答案是 `3670`，因为 SID 仍然有碰撞。

### 3.2 各 split 统计

| Split | 样本数 | 用户数 | 平均历史长度 | 最短历史 | 最长历史 | 历史+目标事件数 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| train | 36259 | 7034 | 3.7700 | 1 | 10 | 172955 |
| valid | 4532 | 2011 | 4.6425 | 1 | 10 | 25572 |
| test | 4533 | 1987 | 4.9565 | 1 | 10 | 27001 |

### 3.3 重复消费现象

`target_item_sid == last_history_item_sid` 的样本数如下：

| Split | 数量 |
| --- | ---: |
| train | 3069 |
| valid | 269 |
| test | 197 |

这说明数据里重复购买/重复消费并不少见。对推荐模型来说，这意味着：

- “继续预测最后一个 item”并不总是错的。
- 只看 exact reward 容易学到短视策略。
- 但层级 reward 又必须非常小心，不然会被“同 prefix 但非同 item”的近似命中带偏。

---

## 4. 训练样本是怎么组织的

### 4.1 样本的基本单位

当前训练样本不是“一位用户一条样本”，而是“一段前缀轨迹一条样本”。

每一行 CSV 至少包含这些字段：

- `user_id`
- `history_item_title`
- `item_title`
- `history_item_id`
- `item_id`
- `history_item_sid`
- `item_sid`

也就是说，一条训练样本表示：

```text
已知用户历史交互序列 -> 预测下一个 item
```

历史长度在当前数据里是 `1~10`。

### 4.2 数据集是如何从原始格式转成 CSV 的

`convert_dataset.py` 会读取：

- `Industrial_and_Scientific.item.json`
- `Industrial_and_Scientific.index.json`
- `Industrial_and_Scientific.{train,valid,test}.inter`

然后把原始 item id 映射成 SID，并写成训练用 CSV。

这个转换阶段不会把整个用户序列压成一条样本；`.inter` 文件本身已经是 prefix trajectory 格式，所以一位用户会贡献多条训练样本。

### 4.3 当前 SFT 是怎么拼样本的

`sft.py` 当前启用了 3 个数据集，然后用 `ConcatDataset` 直接拼在一起：

| 数据集类 | 作用 | 样本数 |
| --- | --- | ---: |
| `SidSFTDataset` | 历史 SID -> 下一个 SID | 36259 |
| `SidItemFeatDataset` | `sid2title` + `title2sid` 对齐任务 | 7316 |
| `FusionSeqRecDataset` | 历史 SID -> 下一个 item title | 36259 |
| 合计 | SFT 总样本数 | 79834 |

其中 `SidItemFeatDataset` 的 7316 来自：

- `3670` 个 `sid2title` 样本
- `3646` 个 `title2sid` 样本

之所以 `title2sid` 不是 3670，是因为 item title 本身有重复。

### 4.4 当前 RL 是怎么拼样本的

`rl.py` 当前启用了 3 个 RL 训练数据集：

| 数据集类 | 作用 | 样本数 |
| --- | --- | ---: |
| `SidDataset` | 历史 SID -> 下一个 SID | 36259 |
| `RLTitle2SidDataset` | `title2sid` + `description2sid` | 6516 |
| `RLSeqTitle2SidDataset` | 标题序列 -> 下一个 SID | 10000 |
| 合计 | RL 总样本数 | 52775 |

`RLTitle2SidDataset` 的 6516 来自：

- `3646` 个 `title2sid`
- `2870` 个 `description2sid`

### 4.5 为什么“只做 SID 推荐任务”通常效果不好

如果只保留 `SidSFTDataset`，模型看到的训练输入输出几乎都是这种形式：

```text
<a_236><b_231><c_226>, <a_42><b_80><c_160>, ...
```

这会带来几个问题：

1. SID 是离散符号，不是天然可解释的语义文本。
2. 如果没有 `title2sid / sid2title / history->title` 这类辅助任务，LLM 很难把 SID 空间和自然语言语义空间绑定起来。
3. SID 推荐任务本身 reward 很稀疏，后续 RL 会更难学。
4. 纯 SID 序列很容易退化成“记住局部共现模式”，而不是理解 item 语义。

当前 `sft.py` 其实已经在用多任务 SFT，原因正是为了缓解这个问题。

---

## 5. 整个模型的数据流

下面是当前仓库真实代码路径对应的完整数据流：

```text
Amazon item metadata
  -> rq/text2emb/amazon_text2emb.py
  -> Industrial_and_Scientific.emb-qwen-td.npy
  -> rq/rqvae.py
  -> rq/models/rqvae.py + rq/models/rq.py + rq/models/vq.py
  -> rq/generate_indices.py
  -> Industrial_and_Scientific.index.json
  -> convert_dataset.py
  -> train/valid/test CSV + info TXT
  -> sft.py
  -> output/final_checkpoint
  -> rl.py + minionerec_trainer.py
  -> rl_output*/final_checkpoint
  -> evaluate.py / calc.py / test.py
  -> final_result_*.json / manual_test_summary.json
```

### 5.1 第一步：文本元数据转 embedding

入口是 `rq/text2emb/amazon_text2emb.py`。

流程如下：

1. 读取 `Industrial_and_Scientific.item.json`
2. 拼接每个 item 的 `title + description`
3. 用冻结的 Qwen `AutoModel` 编码
4. 对 `last_hidden_state` 做带 mask 的 mean pooling
5. 保存成 `Industrial_and_Scientific.emb-qwen-td.npy`

代码上是标准的 masked mean pooling，不是只取最后一个 token，也不是 CLS pooling。

### 5.2 第二步：embedding 量化成 SID

入口是 `rq/rqvae.py`。

流程如下：

1. item embedding 先过 MLP encoder
2. 进入 residual vector quantizer，逐层量化残差
3. 得到多层离散 code
4. decoder 把量化结果重构回原 embedding
5. 用 `reconstruction loss + quantization loss` 训练

最终每个 item 会得到一个三层 SID，例如：

```text
<a_223><b_80><c_165>
```

### 5.3 第三步：SID 写回训练集

入口是 `convert_dataset.py`。

它把交互数据里的原始：

- `history_item_id`
- `item_id`

转换成：

- `history_item_sid`
- `item_sid`

同时保留：

- `history_item_title`
- `item_title`

所以后续 SFT 和 RL 都可以同时用到 SID 空间和自然语言空间。

### 5.4 第四步：SFT

入口是 `sft.py`。

SFT 先做三件事：

1. 从 `index.json` 收集所有新增 SID token
2. 扩展 tokenizer 和 embedding matrix
3. 在多任务数据上做监督微调

当前 Industrial 的 SID token 统计如下：

| 层级 | 理论码本大小 | 实际用到的 token 数 |
| --- | ---: | ---: |
| level-0 | 256 | 48 |
| level-1 | 256 | 256 |
| level-2 | 256 | 256 |
| 合计新增 SID token | - | 560 |

### 5.5 第五步：RL / GRPO

入口是 `rl.py`，训练器主体在 `minionerec_trainer.py`。

RL 流程是：

1. 对同一个 prompt 生成 `G` 个 completion
2. 对每个 completion 计算 reward
3. 在 group 内做均值/方差归一化，得到 advantage
4. 用带 reference KL 的 GRPO loss 更新策略

当前所有实验都配合 constrained decoding，所以测试期离线结果里 `valid_sid_rate = 1.0`，说明生成出来的候选都是合法 SID。

### 5.6 第六步：离线评测

评测链路是：

- `evaluate.py` 负责生成 Top-K SID 候选
- `calc.py` 负责算 HR/NDCG
- `test.py` 负责补充 `exact_match_acc`、`level0_acc`、`mean_hier_score` 等细粒度指标

需要特别注意的一点是：

- `test.py` 用正则正确解析 `<a_x><b_y><c_z>`
- `reward_utils.py` 却没有这么做

这就是当前“训练期层级奖励”和“测试期层级评估”不一致的来源之一。

---

## 6. SID、embedding 与 RQ-VAE 怎么设计

### 6.1 embedding 是怎么拿到的

`rq/text2emb/amazon_text2emb.py` 的实现很直接：

- 编码器：Qwen `AutoModel`
- 输入文本：`title + description`
- 输出表征：`last_hidden_state` 做 masked mean pooling
- 保存格式：`.npy`

这一步没有训练文本编码器，编码器是冻结使用的。

### 6.2 RQ-VAE 结构

`rq/rqvae.py` 和 `rq/models/rqvae.py` 的默认配置是：

| 项目 | 配置 |
| --- | --- |
| 输入维度 | 取决于 embedding 维度，当前 Industrial 是 text encoder 输出维度 |
| encoder/decoder 隐层 | `[2048, 1024, 512, 256, 128, 64]` |
| 量化维度 `e_dim` | `32` |
| 码本层数 | `3` |
| 每层码本大小 | `[256, 256, 256]` |
| commitment beta | `0.25` |
| loss | `reconstruction loss + quantization loss` |

所以当前 SID 的深度是 `3`，每一层的码本大小都是 `256`。

理论容量是：

```text
256 x 256 x 256 = 16,777,216
```

对 `3686` 个 item 来说容量完全足够。

### 6.3 residual quantizer 是怎么做的

`rq/models/rq.py` 里的逻辑是标准 residual quantization：

1. 第一层量化原始 latent
2. 用 residual = latent - quantized_1
3. 第二层量化 residual
4. 再继续残差迭代
5. 最终把各层量化向量相加得到 `x_q`

这就是为什么 SID 可以写成多层路径，而不是单个离散 id。

### 6.4 碰撞怎么处理

当前仓库里有两套思路：

1. `rq/generate_indices.py`
2. `rq/generate_indices_plus.py`

当前实际 Industrial index 的碰撞现象表明，至少最终用到的数据文件并没有完全去碰撞。

`rq/generate_indices.py` 的处理方式是：

- 先正常量化全部 item
- 检查是否有 SID collision
- 对发生碰撞的 item，再用 `use_sk=True` 重新量化
- 最多迭代 20 轮

但它在代码里也明确写了：

```text
There are often duplicate items in the dataset, and we no longer differentiate them
```

也就是说，这个流程不是“必须彻底去碰撞”，而是“尽量减少碰撞，剩下的允许保留”。

当前 Industrial 的实际碰撞统计如下：

| 指标 | 数值 |
| --- | ---: |
| item 总数 | 3686 |
| 唯一 SID 数 | 3670 |
| 额外 collision item 数 | 16 |
| collision group 数 | 15 |
| 最大 collision group size | 3 |

因此，如果汇报“碰撞怎么处理”，最准确的说法是：

- 训练后会对碰撞 item 做多轮重编码尝试；
- 但最终并没有完全消除碰撞；
- 当前 Industrial 仍然保留 16 个额外冲突 item。

### 6.5 一个值得注意的结构现象

一级码本只实际用了 `48` 个 token，而二级、三级都用了 `256` 个。

这说明：

- 最粗粒度的父节点没有充分展开；
- 很多 item 被压到少数一级簇里；
- 如果 reward 强调 prefix 接近，就很容易把模型往“猜对父类”而不是“猜对 item”上推。

---

## 7. SFT 怎么做

### 7.1 当前代码里真正启用的 SFT 任务

`sft.py` 当前启用的是：

1. `SidSFTDataset`
2. `SidItemFeatDataset`
3. `FusionSeqRecDataset`

没有真正启用的是：

- `TitleHistory2SidSFTDataset`
- `PreferenceSFTDataset`
- `UserPreference2sidSFTDataset`

这些类虽然写在 `data.py` 里，但当前默认 SFT 代码没有把它们加进训练集。

### 7.2 当前 SFT 的本质

当前 SFT 并不是只做“历史 SID -> 下一个 SID”，而是三类 supervision 混训：

1. 序列推荐任务：历史 SID -> 下一个 SID
2. 对齐任务：SID <-> title
3. 语义桥接任务：历史 SID -> 下一个 item title

这三类任务拼在一起，目的是把离散 SID token 拉回到 LLM 的自然语言语义空间。

### 7.3 tokenizer 和 embedding 怎么处理

`TokenExtender` 会扫描 `index.json` 里的所有 SID token，并把它们加入 tokenizer。

如果 `freeze_LLM=True`，理论上只训练新增 SID token 的 embedding；但当前实现里有一个潜在 bug：

- `sft.py` 里使用了 `original_vocab_size`
- 但这个变量没有定义

所以 `freeze_LLM=True` 这条路径当前实际上是有风险的。好在当前 `sft.sh` 用的是：

```text
--freeze_LLM False
```

因此这个 bug 没有直接影响你现在已经跑完的结果，但它说明 SFT 代码里还有未走通的分支。

### 7.4 为什么只做 SID 推荐任务效果不好

如果只做 `SidSFTDataset`，会有三个直接问题：

1. 监督信号太单一，只教模型输出离散 SID，不教它理解 SID 对应的 item 语义。
2. 新增的 SID token 是人工离散码，不像自然语言 token 那样有预训练语义。
3. 后续 RL 又主要奖励“命中对的 SID”，会让整个训练信号更稀疏。

所以只做 SID 推荐，模型很容易：

- 学到序列模式；
- 但学不到 item 语义；
- 对层级 reward 和 title/description reward 也接不上。

这也是为什么当前代码已经默认把 `sid2title/title2sid` 和 `history->title` 任务一起混进 SFT。

---

## 8. GRPO / RL 是怎么做的

### 8.1 当前有哪些 reward 变体

`rl.py` 当前支持的主要变体如下：

| 变体 | 实现方式 |
| --- | --- |
| `rule` | exact match 命中给 1，否则 0 |
| `ranking` | `rule_reward + ndcg_rule_reward` |
| `hierarchical` | 按 level0/1/2/exact 分层给分 |
| `hierarchical_ranking` | `hierarchical_rule_reward + hierarchical_ndcg_reward` |
| `pclpo_soft` | hierarchical 基础上，同 parent prefix 再加 bonus |
| `pclpo_margin_ranking` | hierarchical + same-parent margin + rank bonus |
| `semantic` | 用 embedding cosine similarity 打分 |
| `sasrec` | 用协同过滤模型的打分做 reward |

你这次实验真正关注的是中间四种：

- `hierarchical`
- `hierarchical_ranking`
- `pclpo_soft`
- `pclpo_margin_ranking`

### 8.2 当前实现里的 GRPO 目标

`minionerec_trainer.py` 的核心流程是：

1. 对同一个 prompt 生成 `G = num_generations` 个 completion
2. 计算每个 completion 的 reward
3. 按 group 计算：

```text
advantage = (reward - group_mean) / (group_std + 1e-4)
```

4. 再加上对 reference model 的 KL 约束

默认 loss 分支可以理解为：

```text
maximize advantage-weighted policy score
while keeping policy close to reference model
```

对应代码里更具体的形式是：

```text
loss = - (policy_term - beta * KL)
```

其中：

- `policy_term` 用的是 `exp(logpi - stopgrad(logpi)) * advantage`
- 数值上等于 1，但梯度上等价于 policy gradient 的 surrogate
- `KL` 是当前策略和 reference 策略的 per-token KL

### 8.3 `grpo` 的目标到底是什么

如果用一句话说：

> 对每个推荐 prompt，GRPO 希望让“相对更好的 completion”在同组候选中拿到更高概率，同时不要偏离 SFT reference model 太远。

放到推荐问题里，更具体就是：

- 让正确的 next-item SID 在同一组候选里更占优；
- 同时维持生成稳定性；
- 再配合 constrained decoding 保证输出一定是合法 item SID。

### 8.4 为什么说这里只是“推荐版 GRPO”，不是通用 RLHF

因为这里的动作空间不是开放文本，而是受约束的闭集 item SID：

- 候选必须落在合法 SID 前缀树上；
- 生成阶段用 constrained decoding；
- reward 也是围绕 exact hit / prefix hit / same parent 来定义的。

所以它本质上是“面向离散 item space 的 generative recommendation RL”，而不是开放域文本偏好优化。

---

## 9. 为什么当前效果不理想

这一节是最重要的复盘。

### 9.1 一级问题：层级 reward 的 SID 解析是错的

`reward_utils.py` 里：

```python
for sep in ["::", "|", ",", "/", "-", "_"]:
    if sep in x:
        parts = [p.strip() for p in x.split(sep) if p.strip() != ""]
```

而当前 SID 是：

```text
<a_223><b_80><c_165>
```

所以它会被切成：

```text
['<a', '223><b', '80><c', '165>']
```

这意味着：

- level-0 / level-1 / level-2 根本不再对应真实三层 SID；
- `same_parent_bonus` 也不再是真实 parent；
- `pclpo_margin` 里的 parent 判定也会错。

这不是一个“小误差”，而是 reward 语义本身被破坏了。

### 9.2 二级问题：层级打分不是严格 prefix 逻辑

即使把 SID 正确切成三层，当前 `hierarchical_match_score()` 仍然有一个设计问题：

```python
if pred[0] == tgt[0]:
    score += w_l0
if pred[1] == tgt[1]:
    score += w_l1
if pred[2] == tgt[2]:
    score += w_l2
```

这不是严格的树状 prefix 匹配。它允许：

- level-0 不同；
- 但 level-1 或 level-2 恰好相同；
- 仍然拿到分数。

真正的层级奖励更合理的写法通常应该是：

- level-1 匹配必须建立在 level-0 已匹配的前提下；
- level-2 匹配必须建立在 level-0 和 level-1 都匹配的前提下。

否则它学到的是“共享 token id”而不是“共享路径前缀”。

### 9.3 三级问题：`pclpo_*` 没有真正使用 sibling 关系

`rl.py` 里确实构建了：

- `sibling_map`
- `parent_to_children`

但 `pclpo_soft_score()` 和 `pclpo_margin_score()` 实际上并没有用 `sibling_map` 来判定 sibling 或 child，只是在做：

```python
tuple(pred[:parent_depth]) == tuple(tgt[:parent_depth])
```

所以当前的 `pclpo` 更像：

- “同前缀软奖励”

而不是严格意义上的：

- “parent-child listwise preference optimization”

### 9.4 训练期 reward 与测试期层级评估不一致

`test.py` 里用正则：

```python
SID_RE = re.compile(r"<a_(\d+)><b_(\d+)><c_(\d+)>")
```

这说明离线测试的层级指标是按真实三层结构算的。

但训练期 `reward_utils.py` 却不是这样。

这直接造成：

- 训练在优化一套错误的层级结构；
- 测试在评估一套正确的层级结构。

如果最终结果不好，这个不一致本身就足够解释很多问题。

### 9.5 `evaluate.sh` 有明确 bug

你把 GPU 列表从 `0,1,2,3,4,5,6,7` 改成了 `4,5,6,7,8,9`，但 `evaluate.sh` 仍然检查：

```bash
if [[ ! -f "$temp_dir/0.csv" ]]; then
```

而 `split.py` 会按给定的 GPU id 写文件名，所以实际输出是：

```text
4.csv 5.csv 6.csv 7.csv 8.csv 9.csv
```

不是 `0.csv`。

这意味着当前 `evaluate.sh` 在这组 GPU 设置下是逻辑不一致的。已有结果应该不是靠这份脚本无修改直接稳定产出的，或者说它至少存在明显脆弱性。

### 9.6 `run.sh` 的 resume 逻辑和 `rl.py` 不匹配

`run.sh` 会尝试自动检测 resume flag，但当前 `rl.py --help` 没有暴露对应参数。

所以脚本已经自己打印了警告：

- 发现 checkpoint
- 但不知道怎么 resume
- 可能会重头开始

这会导致：

- 训练资源浪费
- 不同 run 的可比性变差
- 你在 `experiments/runs` 里看到很多半成品目录

### 9.7 SFT 改动很小，所以性能变化大概率不在 SFT

你这次对 `sft.py` 的改动只有：

- 注释掉 `group_by_length`
- `report_to=None` 改成 `report_to=\"none\"`

这两个都不是足以导致推荐效果明显下降的主因。

因此，如果效果没超过原版，最该怀疑的是：

- reward 设计实现
- 训练/评测脚本一致性

而不是 SFT。

---

## 10. 面试式问答

这一节直接按你要回答的问题来写。

### 10.1 用的数据 item 数量有多少，交互记录数量是多少

如果汇报 `Amazon Industrial_and_Scientific`，建议这样回答：

- item 数量：`3686`
- 用户数量：`7694`
- train/valid/test 样本数：`36259 / 4532 / 4533`
- 总样本数：`45324`
- 总交互事件数（历史长度 + target 全部累加）：`225528`

如果老师继续问“唯一 SID 数有多少”，可以补一句：

- 当前唯一 SID 数是 `3670`，因为还有 collision。

### 10.2 训练样本怎么组织

答：

---

## 11. 2026-04-15 补充盘点：日志、权重、统一结果目录

这部分是我在当前 workspace 里重新逐个核对日志、`exp_config.json`、`trainer_state.json`、`final_checkpoint` 后补充的结论。

### 11.1 目前确认完整的实验产物在哪里

已经确认“训练完整”的实验有：

- `01_rule`
- `02_hierarchical`
- `03_hierarchical_ranking`
- `04_pclpo_soft`
- `06_pclpo_margin_ranking`

对应位置如下：

| 实验 | 主要目录 | 完整性判断 |
| --- | --- | --- |
| `01_rule` | `experiments/runs/20260315_140310_pclpo_ablation/01_rule` | 有 `final_checkpoint`，且已有 `01_rule_test.json` |
| `02_hierarchical` | `experiments/runs/20260315_140310_pclpo_ablation/02_hierarchical` | 有 `final_checkpoint`，且已有 `02_hierarchical_test.json` |
| `03_hierarchical_ranking` | `rl_output_03_hierarchical_ranking` | `checkpoint-6598` 满足 `global_step=max_steps=6598`，且有 `final_checkpoint` |
| `04_pclpo_soft` | `rl_output_04_pclpo_soft` | `checkpoint-6598` 满足 `global_step=max_steps=6598`，且有 `final_checkpoint` |
| `06_pclpo_margin_ranking` | `rl_output_06_pclpo_margin_ranking` | `checkpoint-6598` 满足 `global_step=max_steps=6598`，且有 `final_checkpoint` |

也就是说，`04` 的确跑完过，只是它不在 `experiments/runs/20260315_140310_pclpo_ablation` 下面，而是在仓库根目录的独立输出目录里。

### 11.2 关于 `05_pclpo_soft_ranking`

我在当前 workspace 内检索了：

- `exp_config.json`
- `trainer_state.json`
- `final_checkpoint/config.json`
- 含 `05` / `pclpo_soft_ranking` 的日志文件

目前没有检出 `05_pclpo_soft_ranking` 的完整产物。

因此截至这次盘点，比较稳妥的说法是：

- `04_pclpo_soft` 明确存在并跑完；
- `05_pclpo_soft_ranking` 在当前 workspace 中未找到对应完整结果；
- 你后续的另一条完整实验是 `06_pclpo_margin_ranking`。

如果你别处还有 `05` 的输出目录，可以再并入统一结果目录。

### 11.3 统一结果目录已经整理到一个地方

为了避免后面到处翻目录，已经新增：

```text
/root/MiniOneRec-prefix/complete_results_20260415
```

这个目录里统一放了：

- `01_rule/`
- `02_hierarchical/`
- `03_hierarchical_ranking/`
- `04_pclpo_soft/`
- `05_pclpo_soft_ranking/`
- `06_pclpo_margin_ranking/`
- `results/`
- `logs/`
- `current_summary.json`
- `test_metrics_summary.json`
- `evaluate_all_complete.sh`
- `summarize_test_metrics.py`

这里大部分是软链接，不会重复复制大权重。

### 11.4 已有离线 test 指标

目前已确认并汇总的 test 指标有：

| 实验 | HR@10 | NDCG@10 | MRR | exact_match_acc |
| --- | ---: | ---: | ---: | ---: |
| `01_rule` | `0.1571` | `0.1150` | `0.1065` | `0.0827` |
| `02_hierarchical` | `0.1496` | `0.1099` | `0.1020` | `0.0790` |

这两组指标现在已经被写入：

```text
/root/MiniOneRec-prefix/complete_results_20260415/test_metrics_summary.json
```

### 11.5 为什么 `03/04/06` 还没补完 evaluate

这次我直接尝试跑过一次：

- `03_hierarchical_ranking`
- `evaluate.py`
- `batch_size=8`
- `num_beams=50`

结果在生成阶段触发了 CUDA OOM。

所以当前更稳妥的做法不是继续硬顶默认参数，而是把一键脚本整理好，用更保守的参数重新跑，例如：

```bash
GPU_ID=5 EVAL_BATCH_SIZE=2 NUM_BEAMS=50 \
bash /root/MiniOneRec-prefix/complete_results_20260415/evaluate_all_complete.sh
```

如果还 OOM，再降到：

```bash
GPU_ID=5 EVAL_BATCH_SIZE=1 NUM_BEAMS=20 \
bash /root/MiniOneRec-prefix/complete_results_20260415/evaluate_all_complete.sh
```

### 11.6 后续推荐工作流

后面建议统一按下面这个入口做：

1. 权重、日志、已有结果先只看 `complete_results_20260415/`
2. 缺 test json 的实验用 `evaluate_all_complete.sh` 续跑
3. 跑完后执行 `summarize_test_metrics.py`
4. 最终只对 `test_metrics_summary.json` 做横向比较

- 每条样本是一段 prefix trajectory，不是一位用户一条样本。
- 一条样本包含历史 item 序列和下一个 target item。
- 训练文件里同时保留 `item_id`、`item_sid`、`item_title` 三种视图。
- SFT 是多任务拼接训练，不是单任务。

### 10.3 SID 训练的流程是什么

答：

1. 用 item 的 `title + description` 生成连续 embedding。
2. 用 RQ-VAE 把 embedding 压成三层离散 code。
3. 把三层 code 写成 SID，例如 `<a_223><b_80><c_165>`。
4. 再把交互数据里的 item id 替换成 SID，生成 SFT/RL 用的训练集。
5. SFT 阶段让 LLM 学会从历史生成下一个 SID。
6. RL 阶段再用 GRPO 在 constrained decoding 下细化排序行为。

### 10.4 embedding 怎么获取

答：

- 使用冻结的 Qwen text encoder。
- 输入是 `title + description`。
- 取 `last_hidden_state`，按 attention mask 做 mean pooling。
- 输出保存为 `.npy`，供 RQ-VAE 使用。

### 10.5 RQ-VAE 怎么设计，码本大小和深度是多少

答：

- encoder/decoder 是 MLP
- 隐层是 `[2048, 1024, 512, 256, 128, 64]`
- 量化维度 `e_dim=32`
- 量化器深度是 `3`
- 每层码本大小是 `256`

也就是：

```text
3-level residual quantization
codebook size = 256 per level
depth = 3
```

### 10.6 碰撞怎么处理

答：

- 先正常生成 SID
- 对发生 collision 的 item 再用 Sinkhorn 约束重新量化
- 最多迭代 20 轮
- 但当前并没有完全消除碰撞

当前 Industrial 最终仍有：

- `16` 个额外 collision item
- `15` 个 collision group

### 10.7 SFT 怎么做

答：

- 先扩展 tokenizer，把 SID token 加进去
- 然后把三类样本拼起来做多任务 SFT：
  - 历史 SID -> 下一个 SID
  - SID <-> title
  - 历史 SID -> 下一个 title

当前 SFT 总样本数是 `79834`。

### 10.8 为什么只做 SID 推荐任务效果不好

答：

- 纯 SID 是离散符号，没有自然语言语义；
- 只做 SID 推荐，相当于只做“序列模式拟合”；
- 不做 `sid2title/title2sid` 等辅助任务，模型很难把 SID 和 item 语义对齐；
- 后续 RL 又主要看 SID 命中，reward 很稀疏，会进一步放大这个问题。

### 10.9 GRPO 的目标是什么

答：

- 对同一个推荐 prompt 生成多个候选 item SID；
- 在同组候选内做 reward 归一化；
- 提升相对高 reward completion 的概率；
- 同时用 KL 约束让策略不要偏离 reference model 太远。

如果一句话概括：

> GRPO 的目标是在合法 item SID 空间里，把“更可能是正确下一个 item”的候选相对提上去，而不是无约束地生成开放文本。

---

## 11. 最后结论

如果你要对外汇报这份工作，我建议最稳的口径是：

1. 你相对原版的主要创新在 RL reward，而不是 SFT 或 SID 生成。
2. 当前已经完成的离线证据表明，`hierarchical` 没有超过 `rule`。
3. 造成结果不理想的最大嫌疑，不是想法本身，而是 reward 实现和评测链路存在不一致。
4. 下一步最值得做的不是继续堆更多 reward 变体，而是先修：
   - SID 层级解析
   - prefix 级联打分
   - sibling/parent 真实使用
   - 统一离线评测

在这些基础问题没修之前，任何“新 reward 比旧 reward 强/弱”的结论都不够稳。
