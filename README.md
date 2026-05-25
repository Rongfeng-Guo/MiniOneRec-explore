# MiniOneRec Prefix and Multihead Variants

This repository is a cleaned research-code release containing two MiniOneRec-based experimental variants:

- `prefix/`: a prefix-oriented variant
- `multihead/`: a multi-head variant

The goal of this repository is to make the core implementation easier to inspect and reuse without bundling large private training artifacts.

## Included in This Release

- training and evaluation code
- dataset conversion and preprocessing scripts
- SID construction utilities under `rq/`
- variant-specific shell entrypoints
- lightweight documentation and illustrative assets

## Excluded from This Release

To keep the repository suitable for GitHub distribution, the following are intentionally excluded:

- model checkpoints and weight files
- generated experiment outputs such as `output/`, `outputs/`, `results/`, and `experiments/`
- RL run artifacts such as `rl_output*`
- duplicated local datasets and generated index dumps
- cache files and temporary files

## Repository Structure

```text
.
├── prefix/
│   ├── README.md
│   ├── sft.py / rl.py / evaluate.py
│   ├── convert_dataset.py
│   ├── data/
│   ├── rq/
│   └── ...
└── multihead/
    ├── README.md
    ├── sft.py / rl.py / evaluate.py
    ├── convert_dataset.py
    ├── data/
    ├── rq/
    └── ...
```

Each subdirectory is largely self-contained so the two variants can be reviewed independently.

## Variant Overview

### `prefix/`

The `prefix/` branch contains a prefix-oriented MiniOneRec variant together with its training, RL, and evaluation scripts.

Useful entrypoints:

- `prefix/sft.sh`
- `prefix/rl.sh`
- `prefix/evaluate.sh`
- `prefix/run.sh`

### `multihead/`

The `multihead/` branch contains a multi-head MiniOneRec variant with its own training and experiment scripts.

Useful entrypoints:

- `multihead/sft.sh`
- `multihead/rl.sh`
- `multihead/evaluate.sh`
- `multihead/run_all_experiments.sh`

## Quick Start

The exact runtime environment depends on your local datasets and model checkpoints, but the typical workflow is:

```bash
# 1. create environment
conda create -n minionerec python=3.11 -y
conda activate minionerec

# 2. install dependencies for a variant
pip install -r prefix/requirements.txt
# or
pip install -r multihead/requirements.txt

# 3. prepare data and SID indices
# see convert_dataset.py, data/, and rq/

# 4. run training / evaluation
bash prefix/sft.sh
bash prefix/rl.sh
bash prefix/evaluate.sh
```

If you are using the multi-head branch, replace the commands with the corresponding scripts under `multihead/`.

## Notes on Reproducibility

- Some scripts still assume local dataset layouts or environment-specific paths.
- This repository is intended as a cleaned code release, not a fully packaged benchmark reproduction bundle.
- To make the project fully reproducible, the next practical steps would be standardizing configs, dataset paths, and end-to-end experiment entrypoints.

## Reading Order

If you are new to this repository, the fastest way to understand it is:

1. Read this root README.
2. Read `prefix/README.md` and `multihead/README.md` for branch-specific context.
3. Inspect `sft.py`, `rl.py`, and `evaluate.py` in the variant you care about.
4. Check `convert_dataset.py`, `data/`, and `rq/` for preprocessing and SID construction.

## License

Please refer to the license files included in each variant directory.
