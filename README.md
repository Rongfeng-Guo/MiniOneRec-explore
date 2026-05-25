# MiniOneRec Prefix and Multihead Variants

This repository packages two research code variants derived from local MiniOneRec experiments:

- `prefix/`: the prefix-tuning oriented variant
- `multihead/`: the multi-head variant

## Notes

- This repository intentionally excludes large checkpoints, training outputs, experiment artifacts, and generated result bundles.
- The goal is to preserve runnable source code, scripts, configs, and lightweight documentation for sharing and follow-up development.

## Structure

- `prefix/`
- `multihead/`

## Excluded Artifacts

The original working directories on the server contained large folders such as:

- `output/`
- `outputs/`
- `results/`
- `experiments/`
- `rl_output*`
- model checkpoints and weight files

Those are not included here.
