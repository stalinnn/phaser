#!/bin/bash
# Official Reproduction Script for TGN (Fig 21 & Fig 22)
# Usage: bash run_reproduction.sh

# 1. Environment Setup
export HF_ENDPOINT=https://hf-mirror.com  # Optional: For CN region
export OMP_NUM_THREADS=4

# 2. Experiment 1: WikiText-103 Convergence (Fig 21)
# Demonstrates "Hysteresis Awakening" and convergence to ~9% sparsity.
# Requires: 6x A100/A800 (80GB)
echo ">>> [1/2] Running WikiText-103 Convergence Experiment..."
torchrun --nproc_per_node=6 experiments/fig21_wikitext/train.py \
    --model_size small \
    --dataset wikitext \
    --batch_size 16 \
    --epochs 10 \
    --lr 6e-4 \
    --sparsity_penalty 0.002 \
    --out_dir result_reproduce/fig21

# 3. Experiment 2: Early Dynamics SOTA Battle (Fig 22)
# Compares Mamba vs Transformer vs TGN (GRU)
echo ">>> [2/2] Running SOTA Battle..."
mkdir -p result_reproduce/fig22

# 3.1 Mamba (Reference)
torchrun --nproc_per_node=6 experiments/fig22_sota_battle/train.py \
    --model_type mamba \
    --model_size medium \
    --out_dir result_reproduce/fig22

# 3.2 Transformer (Baseline)
torchrun --nproc_per_node=6 experiments/fig22_sota_battle/train.py \
    --model_type transformer \
    --model_size medium \
    --out_dir result_reproduce/fig22

# 3.3 TGN (Ours)
torchrun --nproc_per_node=6 experiments/fig22_sota_battle/train.py \
    --model_type tgn \
    --model_size medium \
    --out_dir result_reproduce/fig22

echo "✅ All reproduction experiments finished. Check 'result_reproduce/' for logs."
