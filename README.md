# HRAC

History-Residual Adaptive Clipping (HRAC) is a Byzantine-robust aggregation method for federated learning experiments under label-flipping and model-poisoning attacks.

This repository contains the experiment code used to compare HRAC with common robust aggregators such as mean, trimmed mean, FABA, centered clipping, and LFighter.

## What Is Included

- `ByrdLab/`: core training, attack, aggregation, task, and dataset utilities.
- `main CMomentum.py`: main experiment entry point for momentum-based centralized federated learning.
- `run.ps1`: PowerShell wrapper for running `main CMomentum.py` with the configured local Python interpreter.
- `run_cifar10_b3_all.py`: batch runner for CIFAR-10 experiments over attacks, data partitions, and aggregators.
- `draw_fig/`: plotting scripts for generated experiment records.
- `argsParser.py`: command-line options shared by the experiment scripts.

Local papers, proof drafts, external repositories, datasets, and generated records are intentionally ignored by Git.

## Aggregators

The main script supports:

- `mean`
- `trimmed-mean`
- `faba`
- `cc`
- `lfighter`
- `hrac`

HRAC adds history-aware residual clipping, adaptive norm/change statistics, and optional ablation modes.

## Attacks

Common attack options include:

- `none`
- `label_flipping`
- `msa`
- `hismsa`
- `ipm`
- `alie`
- `poisonedfl`

Exact behavior is implemented in `ByrdLab/attack.py` and selected in `main CMomentum.py`.

## Setup

Install the Python dependencies used by the original experiment code:

- Python 3.8
- PyTorch
- matplotlib
- networkx
- numpy
- scikit-learn

Download datasets locally into `dataset/`. Generated logs and results are written under `record/`. Both folders are ignored by Git.

## Run A Single Experiment

```bash
python "main CMomentum.py" --aggregation hrac --attack label_flipping --data-partition iid
```

Example with a non-IID split:

```bash
python "main CMomentum.py" --aggregation hrac --attack hismsa --data-partition noniid
```

On Windows, the wrapper can be used:

```powershell
.\run.ps1 --aggregation hrac --attack label_flipping --data-partition iid
```

## HRAC Ablations

HRAC ablation options are available through:

```bash
python "main CMomentum.py" --aggregation hrac --hrac-ablation-experiment --hrac-ablation no_nu_weighting
```

Supported ablation names:

- `full`
- `no_global_cap`
- `no_residual_clip`
- `no_nu_weighting`
- `global_cap_only`

Additional HRAC parameters:

- `--hrac-rho-b`
- `--hrac-rho-mu`
- `--hrac-rho-g`

## Run The CIFAR-10 Sweep

```bash
python run_cifar10_b3_all.py
```

This runs combinations of attacks, IID/non-IID partitions, and aggregators defined inside the script.

## Notes

This repository is kept focused on source code. The following are not uploaded:

- local paper files and `.tex` drafts
- generated datasets and records
- third-party repositories kept locally
- temporary test files
- external experiment folders
