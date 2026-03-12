#!/bin/bash

# Cloud Execution Script for TGN Ablation Study
# Usage: bash run_cloud_ablation.sh

echo ">>> Setting up environment..."

# 1. Install dependencies if missing (assuming standard PyTorch image)
pip install matplotlib requests tqdm

# 2. Set permissions
chmod +x code/experiment_ablation_associative.py

# 3. Check GPU count
GPU_COUNT=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)
echo ">>> Detected $GPU_COUNT GPUs."

# 4. Run DDP Experiment
# We use MASTER_PORT=29500 (default) or random to avoid conflicts
export MASTER_PORT=$(shuf -i 20000-65000 -n 1)

echo ">>> Starting Distributed Training..."
torchrun --nproc_per_node=$GPU_COUNT code/experiment_ablation_associative.py

echo ">>> Done! Results saved in figures/ablation_associative.png"
