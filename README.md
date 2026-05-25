# MiniOneRec Prefix and Multihead Variants

This repository contains two lightweight research code variants built from local `MiniOneRec` experiments:

- `prefix/`: a prefix-oriented variant
- `multihead/`: a multi-head variant

The repository is organized as a cleaned, shareable code release rather than a full training workspace.

## What Is Included

- source code
- training and evaluation scripts
- configuration files
- lightweight documentation
- small illustrative assets in the top-level `assets/` folders

## What Is Excluded

To keep the repository practical for GitHub, the following artifacts are intentionally excluded:

- model checkpoints and weight files
- `output/`, `outputs/`, `results/`
- `experiments/`
- `rl_output*`
- cached files and temporary files
- duplicated local dataset dumps and generated index artifacts

## Repository Layout

```text
.
├── prefix/
└── multihead/
```

Each subdirectory keeps its own scripts and code structure so the two variants can be inspected independently.

## Notes

- This repository reflects a cleaned snapshot of the codebase used on an internal server.
- Some scripts may still expect local datasets or environment-specific paths that are not bundled here.
- If you want to make the project fully reproducible, the next step would be to standardize data preparation, dependency setup, and experiment entrypoints.

## Suggested Next Improvements

- add a top-level environment setup guide
- unify duplicated utility modules across variants
- document the exact experimental differences between `prefix` and `multihead`
- add minimal example commands for training and evaluation
