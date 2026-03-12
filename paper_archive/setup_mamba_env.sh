#!/bin/bash

# Create new environment
conda create -n mamba_env python=3.10 -y
source activate mamba_env

# Install PyTorch 2.1.0 + CUDA 11.8 (Most stable for Mamba)
pip install torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 --index-url https://download.pytorch.org/whl/cu118

# Install packaging
pip install packaging

# Install Mamba dependencies
pip install causal-conv1d>=1.0.0
pip install mamba-ssm>=1.0.1

# Install other deps
pip install datasets tiktoken tqdm transformers matplotlib pandas

echo "=========================================="
echo "✅ Environment 'mamba_env' created!"
echo "👉 Run: 'conda activate mamba_env' to start."
echo "=========================================="
