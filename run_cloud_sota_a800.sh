#!/bin/bash

# 激活环境
source activate mamba_gold || conda activate mamba_gold

# 设置 HuggingFace 镜像 (国内集群加速)
export HF_ENDPOINT=https://hf-mirror.com

# 显卡数量
GPUS=6
# 每个GPU的Batch Size (A800 80G 可以设大一点，比如 32 或 64)
BATCH_SIZE=32

echo "======================================================="
echo "🚀 Starting SOTA Battle on $GPUS x A800 GPUs"
echo "======================================================="

# Ensure output directory exists BEFORE logging starts
mkdir -p result_sota_a800

# 1. Run Mamba (The Challenger)
echo ">>> Running Mamba..."
torchrun --nproc_per_node=$GPUS run_cloud_sota_battle.py \
    --model_type mamba \
    --model_size medium \
    --out_dir result_sota_a800 \
    2>&1 | tee result_sota_a800/mamba.log

# 2. Run Transformer (The Baseline)
echo ">>> Running Transformer..."
torchrun --nproc_per_node=$GPUS run_cloud_sota_battle.py \
    --model_type transformer \
    --model_size medium \
    --out_dir result_sota_a800 \
    2>&1 | tee result_sota_a800/transformer.log

# 3. Run TGN (Our Model)
echo ">>> Running TGN..."
torchrun --nproc_per_node=$GPUS run_cloud_sota_battle.py \
    --model_type tgn \
    --model_size medium \
    --out_dir result_sota_a800 \
    2>&1 | tee result_sota_a800/tgn.log

echo "✅ All Done! Results are in result_sota_a800/"
