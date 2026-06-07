# GitHub Import Notes

Maintenance date: `2026-06-08`

## Mapping

| Item | Value |
| --- | --- |
| Cleaned repository path | `/home/grf/minionerec_prefix_multihead_repo` |
| GitHub repository | `git@github.com:Rongfeng-Guo/MiniOneRec-explore.git` |
| Previous repository URL | `git@github.com:Rongfeng-Guo/MiniOneRec-prefix-multihead-code.git` |
| Branch | `main` |
| Initial import commit | `a9744b1` |
| Current base before this documentation pass | `385f43c` |

## Source Branches

This cleaned repository combines three heavier local workspaces:

| Cleaned directory | Original workspace |
| --- | --- |
| `prefix/` | `/home/grf/migrate_from_a100_20260511/MiniOneRec-prefix` |
| `multihead/` | `/home/grf/migrate_from_a100_20260511/MiniOneRec-multihead` |
| `consistency/` | `/home/grf/migrate_from_a100_20260511/MiniOneRec-consistency` |

The original workspaces contain large model checkpoints and run outputs. This
repository is the lightweight code-and-documentation version.

## What This Repository Contains

Included:

- variant source code for `prefix/`, `multihead/`, and `consistency/`
- preprocessing and SID construction code required to inspect the branches
- branch-specific README files and experiment notes
- lightweight assets needed by the inherited README material
- `compare.md` ledgers for the prefix and multihead branches

Excluded:

- model checkpoints and tokenizer payloads
- full `output/`, `outputs/`, `results/`, `experiments/`, and `rl_output*`
  directories
- raw Amazon data and generated preprocessing artifacts
- caches, logs, and local runtime files

## Maintenance Rules

1. Keep the root README focused on cross-branch motivation, evidence, and the
   recommended reading order.
2. Keep branch-specific details inside each branch README.
3. Commit compact Markdown summaries when results change.
4. Keep raw prediction dumps, checkpoints, run logs, and generated data local.
5. Before pushing, check:

```bash
git status --short
git diff --cached --stat
git ls-files -z | xargs -0 du -h | sort -h | tail
```

## Current Interpretation

The cleaned repository is useful as a negative/diagnostic research record:

- the variants are technically meaningful;
- they identify concrete objective-drift and hierarchy-quality issues;
- they do not yet prove a stable improvement over the strongest local baseline.

The next useful experiment would be a controlled ablation that restores the
original mixed-task MiniOneRec SFT base and adds each structural idea one at a
time.
