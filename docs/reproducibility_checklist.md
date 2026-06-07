# Reproducibility Checklist

This checklist is intended for future MiniOneRec-explore runs. The repository
contains several research branches that change different parts of the training
problem, so every reported result should make the comparison boundary explicit.

## Run Identity

- Branch: `prefix`, `multihead`, or `consistency`
- Commit hash:
- Date:
- Machine / GPU:
- Python / CUDA / PyTorch versions:
- Random seed:

## Data and SID Setup

- Dataset name and split:
- Number of train / validation / test samples:
- SID depth:
- Codebook size per level:
- SID collision rate, if measured:
- Text encoder or item embedding source:
- Candidate filtering and constrained decoding rules:

## Training Setup

- SFT checkpoint source:
- Whether the SFT base matches the original MiniOneRec mixed-task setup:
- RL checkpoint source, if any:
- Main script:
- Full command:
- Important hyperparameters:
- Loss terms and their weights:

## Evaluation Protocol

- Evaluation script:
- Test set size:
- Metrics reported:
- Top-K candidate space:
- Decoding mode:
- Exact-match definition:
- Output directory:

## Interpretation Notes

- Is this a strict single-variable comparison?
- Which part of the original task changed?
- Which result is the strongest negative or positive evidence?
- What follow-up run would falsify the current interpretation?
