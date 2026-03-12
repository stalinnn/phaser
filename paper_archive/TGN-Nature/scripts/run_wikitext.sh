#!/bin/bash
#SBATCH --job-name=tgn_scaling
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=6
#SBATCH --gres=gpu:6
#SBATCH --time=48:00:00
#SBATCH --output=tgn_scaling_%j.out

# Load modules (adjust based on your cluster)
# module load cuda/12.1 cudnn/8.9 python/3.10  # Assuming these are available or use container

# Use HF Mirror for stability in CN region
export HF_ENDPOINT=https://hf-mirror.com

# Virtual env (if needed)
# source venv/bin/activate

# Install dependencies if needed
# pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
# pip install tiktoken datasets tqdm wandb

# Run Distributed Training
# 1. WikiText-103 (Standard Benchmark)
# Size: Small (125M) to validate mechanics
# Using 6 GPUs, reduced batch size to fit memory (effective batch still large via 6 GPUs)
torchrun --nproc_per_node=6 run_cloud_scaling_experiment.py \
    --model_size small \
    --dataset wikitext \
    --batch_size 16 \
    --epochs 10 \
    --out_dir ./result/wikitext_small

# 2. WikiText-103 (Scale Up)
# Size: Medium (350M) to validate Gate Rate behavior at scale
torchrun --nproc_per_node=6 run_cloud_scaling_experiment.py \
    --model_size medium \
    --dataset wikitext \
    --batch_size 8 \
    --epochs 5 \
    --out_dir ./result/wikitext_medium
