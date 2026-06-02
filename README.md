# MiniOneRec Prefix, Multihead, and Consistency Variants

This repository is a cleaned research-code release for three MiniOneRec-based experimental branches that were developed to test whether stronger structure on SID prediction can improve generative recommendation:

- `prefix/`: a prefix-oriented branch
- `multihead/`: a multi-head branch with hierarchical auxiliary supervision
- `consistency/`: a multi-view consistency branch that aligns SID-view and title-view SFT

Unlike a generic code dump, this repository is meant to document the actual research path: what we changed, why we changed it, what phenomena we observed, and why the current variants did not yet produce a stable win over the strongest local baseline.

## Motivation

MiniOneRec maps each item into a 3-level SID and trains a generative recommender to predict the next SID from user history. That design naturally suggests two research questions:

1. Can we make the model use the hierarchical structure inside the SID more explicitly?
2. Can coarse-to-fine supervision improve recommendation quality instead of only predicting the final exact SID token sequence?

The three branches in this repository were created to explore those questions from different angles.

## What We Tried

### `multihead/`

The `multihead/` branch adds explicit level-wise supervision on the 3-level SID.

The core idea is:

- keep the next-SID generation task;
- split the target SID into level-0, level-1, and level-2 labels;
- add three auxiliary classification heads during SFT;
- continue with RL after SFT.

In code terms, the important changes are concentrated in:

- `multihead/data.py`
- `multihead/sft.py`
- `multihead/rl.py`
- `multihead/minionerec_trainer.py`

### `prefix/`

The `prefix/` branch explores reward shaping and prefix-aware structure during RL. It contains several experiments around hierarchical and ranking-style rewards, including more aggressive variants that try to give partial credit when the predicted SID matches the target only at coarse levels.

In practice, this branch is where we tested whether a better reward design can translate coarse SID-level correctness into better exact next-item recommendation.

### `consistency/`

The `consistency/` branch explores a paired multi-view SFT setup. For each training row it constructs:

- `SID history -> next SID`
- `title history -> next SID`

The branch then trains both views jointly and adds consistency constraints between their prediction distributions and anchor representations.

In code terms, the most important additions are concentrated in:

- `consistency/sft_mv.py`
- `consistency/paired_mv_dataset.py`
- `consistency/mv_consistency_trainer.py`

## Main Observation

The most important high-level finding is that these variants are not just small add-ons on top of the original MiniOneRec baseline.

Instead, the experimental branches changed part of the base training problem itself.

For example, in the `multihead/` branch:

- the original mixed-task SFT setup is narrowed to a single `SidSFTDataset` task;
- the auxiliary heads are attached to one shared prompt-boundary hidden state;
- RL is then run on top of that modified SFT checkpoint.

This matters because a result like "multihead did not beat baseline" is not the same as saying "adding one lightweight hierarchical regularizer to the original MiniOneRec hurts." In our local implementation, the base objective already drifted.

## Current Results

The most reliable directly comparable local results use the same `4533`-sample test split.

| Variant | Branch | Samples | Exact@1 | HR@10 | NDCG@10 | HR@20 | NDCG@20 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `rl_base` | `multihead` | 4533 | 0.07831 | 0.15442 | 0.11198 | 0.19347 | 0.12180 |
| `rl_aux` | `multihead` | 4533 | 0.07721 | 0.15773 | 0.11233 | 0.18840 | 0.12008 |

What this means:

- `rl_aux` gives a very small gain on `HR@10` and `NDCG@10`;
- but it drops on `Exact@1`, `HR@20`, and `NDCG@20`;
- overall, there is no stable or convincing improvement.

A more detailed pairwise comparison in the experiment summary shows:

- `both_correct = 308`
- `base_only_correct = 47`
- `aux_only_correct = 42`
- `same_prediction_rate = 0.48246`

So the auxiliary branch is not consistently stronger. It changes a small subset of cases, but does not create a clear overall win.

## Phenomena We Observed

Several recurring phenomena showed up during analysis.

### 1. Coarse correctness is easier than exact correctness

Some reward or auxiliary designs improve coarse SID-level agreement more easily than exact item recovery. In other words, the model can become better at predicting the right region of the SID space without actually ranking the correct item higher.

### 2. Structural supervision can change the optimization target a lot

The 3 auxiliary level losses are not a harmless add-on. Once their scale is nontrivial, they reshape what the model is optimizing for during SFT.

### 3. Reward shaping is easy to overinterpret

In the `prefix/` branch, some reward variants looked promising from training dynamics alone, but training-time reward, KL, or diversity metrics were not enough to conclude that final offline Top-K performance had improved.

### 4. SID quality itself is a bottleneck

Our local SID statistics already show structural limitations:

- `3686` index entries
- `3670` unique SIDs
- `16` duplicated entries
- `15` collision groups
- only `48` unique level-0 tokens actually used

That means the hierarchy is informative, but not perfectly clean or uniformly utilized.

## Why We Think the Current Variants Did Not Clearly Surpass the Baseline

The current best explanation is not a single bug, but a combination of design drift and objective mismatch.

### 1. The SFT base changed

In the original MiniOneRec pipeline, SFT is not only `history SID -> next SID`. It mixes multiple tasks that align SID space, item text, and sequence semantics. The `multihead/` branch narrowed that setup substantially.

### 2. The three level heads share one hidden state

In the current `multihead` implementation, level-0, level-1, and level-2 predictions are all made from one prompt-boundary hidden state, rather than from the actual autoregressive positions of the three SID tokens.

That makes the auxiliary objective structurally different from the true generation process.

### 3. Partial-credit rewards do not automatically improve ranking

A reward that values hierarchical overlap can push predictions toward the right SID prefix, but that does not guarantee better exact next-item ranking.

### 4. The hierarchy itself is imperfect

SID collisions and low utilization at the coarsest level limit how much clean supervision the hierarchy can provide.

## What This Repository Contains

This repository keeps the code needed to inspect and continue these experiments, while excluding large private training artifacts.

Included:

- training and evaluation code
- preprocessing and dataset conversion scripts
- SID construction utilities under `rq/`
- variant-specific experiment scripts
- lightweight documentation and illustrative assets

Excluded:

- checkpoints and model weights
- generated outputs such as `output/`, `outputs/`, `results/`, and `experiments/`
- RL artifact directories such as `rl_output*`
- duplicated local datasets and cached files

## Repository Layout

```text
.
├── consistency/
│   ├── README.md
│   ├── sft_mv.py
│   ├── paired_mv_dataset.py
│   ├── mv_consistency_trainer.py
│   ├── data/
│   ├── rq/
│   └── ...
├── prefix/
│   ├── README.md
│   ├── compare.md
│   ├── sft.py / rl.py / evaluate.py
│   ├── convert_dataset.py
│   ├── data/
│   ├── rq/
│   └── ...
└── multihead/
    ├── README.md
    ├── compare.md
    ├── sft.py / rl.py / evaluate.py
    ├── convert_dataset.py
    ├── data/
    ├── rq/
    └── ...
```

## Where To Read First

If you want the fastest route to understanding the work, read in this order:

1. this root `README.md`
2. `multihead/compare.md`
3. `prefix/compare.md`
4. `consistency/sft_mv.py`
5. `consistency/paired_mv_dataset.py`
6. `multihead/sft.py`
7. `multihead/rl.py`
8. `prefix/rl.py`

## Recommended Next Steps

If the goal is to turn these ideas into a stronger paper-quality result, the most reasonable next steps are:

1. restore a truly comparable base that matches the original mixed-task SFT setup;
2. attach hierarchical supervision to the real autoregressive SID positions instead of one shared hidden state;
3. make auxiliary losses warm up gradually instead of competing with AR loss from the start;
4. evaluate reward variants only with unified offline Top-K testing, not just training logs;
5. improve SID quality first by reducing collisions and increasing useful level-0 utilization.

## Quick Start

The exact runtime environment depends on your local datasets and model checkpoints, but a typical workflow is:

```bash
conda create -n minionerec python=3.11 -y
conda activate minionerec

pip install -r prefix/requirements.txt
# or
pip install -r multihead/requirements.txt
# or
pip install -r consistency/requirements.txt

# prepare data and SID indices
# see convert_dataset.py, data/, and rq/

bash multihead/sft.sh
bash multihead/rl.sh
bash multihead/evaluate.sh
```

Use the corresponding scripts under `prefix/` if you are reproducing the prefix-oriented branch.

For the multi-view consistency experiments, the main entry point is `consistency/sft_mv.py`, together with `consistency/paired_mv_dataset.py` and `consistency/mv_consistency_trainer.py`.

## License

Please refer to the license files included in each variant directory.
