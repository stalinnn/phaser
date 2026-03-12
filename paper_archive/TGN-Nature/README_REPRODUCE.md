# Reproduction Guide

This repository contains all the code and scripts necessary to reproduce the experimental results presented in the paper **"Thermodynamic Gated Networks (TGN): Attention as a Geometric Anti-Dissipative Force"**.

## Prerequisites

- **Hardware**: We recommend at least **6x A100 (80GB)** GPUs for full reproduction of the large-scale language modeling experiments (Fig 21 & 22). Smaller experiments (Fig 5, 8, 14) can be run on a single consumer GPU (e.g., RTX 4090).
- **Software**: PyTorch 2.0+, CUDA 11.8+, `mamba_ssm` (optional, for Fig 22 comparison).

## Quick Start (Cloud Cluster)

If you have a SLURM cluster or a multi-GPU machine, you can run all major experiments with a single script:

```bash
cd TGN-Nature
bash scripts/run_reproduction.sh
```

This will sequentially run:
1.  **WikiText-103 Convergence**: Training a 125M TGN model to convergence (Fig 21).
2.  **SOTA Battle**: Comparing early training dynamics of Mamba, Transformer, and TGN (Fig 22).

## Experiment Details

### 1. Physics Simulations (Fig 5)
*   **Path**: `experiments/fig5_spin_glass/`
*   **Command**: `python experiments/fig5_spin_glass/run.py`
*   **Description**: Simulates the 3D Edwards-Anderson spin glass model to demonstrate the "Geometric Tunneling" effect vs Parallel Tempering.

### 2. LLM Rank Dynamics (Fig 8)
*   **Path**: `experiments/fig8_llm_rank/`
*   **Command**: `python experiments/fig8_llm_rank/run.py`
*   **Description**: Probes the singular value spectrum of GPT-2 layers to visualize "Rank Collapse" and subsequent restoration.

### 3. fMRI Analysis (Fig 14)
*   **Path**: `experiments/fig14_fmri/`
*   **Command**: `python experiments/fig14_fmri/run.py`
*   **Description**: Analyzes human fMRI data (OpenNeuro ds003171) to show the correlation between consciousness states and geometric rank.

### 4. Language Modeling at Scale (Fig 21)
*   **Path**: `experiments/fig21_wikitext/`
*   **Command**: See `scripts/run_reproduction.sh`
*   **Description**: Trains a TGN on WikiText-103 to show the spontaneous emergence of ~9% gate sparsity.

### 5. SOTA Comparison (Fig 22)
*   **Path**: `experiments/fig22_sota_battle/`
*   **Command**: See `scripts/run_reproduction.sh`
*   **Description**: Compares TGN against Mamba (SSM) and Transformer on convergence speed and memory capacity.
