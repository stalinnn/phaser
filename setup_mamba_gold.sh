#!/bin/bash

# 1. Kill any existing env to be safe
conda deactivate
conda env remove -n mamba_gold -y

# 2. Create fresh environment
conda create -n mamba_gold python=3.10 "numpy<2.0" -y
source activate mamba_gold

# 3. Install PyTorch 2.1.1 + CUDA 11.8 (The Gold Standard for Mamba 1.x)
# Mamba 1.x is much easier to install than 2.x
pip install torch==2.1.1 torchvision==0.16.1 torchaudio==2.1.1 --index-url https://download.pytorch.org/whl/cu118

# 4. Install build tools
pip install packaging ninja wheel

# 5. Install Mamba dependencies (From Local Uploaded Wheels)
# ------------------------------------------------------------------
# INSTRUCTIONS:
# 1. Download these two files on your local computer:
#    - causal_conv1d-1.2.0.post2+cu118torch2.1cxx11abiFALSE-cp310-cp310-linux_x86_64.whl
#    - mamba_ssm-1.2.0.post1+cu118torch2.1cxx11abiFALSE-cp310-cp310-linux_x86_64.whl
# 2. Upload them to this directory on the cloud server.
# ------------------------------------------------------------------

echo "Looking for local wheel files..."

if [ -f "causal_conv1d-1.2.0.post2+cu118torch2.1cxx11abiFALSE-cp310-cp310-linux_x86_64.whl" ]; then
    echo "✅ Found local causal-conv1d wheel. Installing..."
    pip install causal_conv1d-1.2.0.post2+cu118torch2.1cxx11abiFALSE-cp310-cp310-linux_x86_64.whl
else
    echo "⚠️ Local causal-conv1d wheel not found. Please upload it."
fi

if [ -f "mamba_ssm-1.2.0.post1+cu118torch2.1cxx11abiFALSE-cp310-cp310-linux_x86_64.whl" ]; then
    echo "✅ Found local mamba-ssm wheel. Installing..."
    # Install dependencies FIRST to prevent pip from upgrading to incompatible versions
    pip install "transformers==4.29.0" "numpy<2.0"
    pip install mamba_ssm-1.2.0.post1+cu118torch2.1cxx11abiFALSE-cp310-cp310-linux_x86_64.whl
else
    echo "⚠️ Local mamba-ssm wheel not found. Please upload it."
fi

# Fallback check
python -c "import causal_conv1d; import mamba_ssm; print('✅ Mamba install verified!')" || echo "❌ Mamba install FAILED. Please upload the .whl files."

# 6. Install other deps
# transformers<4.30.0 for compatibility with Torch 2.1's pytree behavior
# numpy<2.0 for ABI compatibility
pip install "transformers<4.30.0" datasets tiktoken tqdm matplotlib pandas "numpy<2.0"

# 7. Verification
python -c "import torch; print(f'Torch: {torch.__version__}, CUDA: {torch.version.cuda}'); import mamba_ssm; print(f'Mamba: {mamba_ssm.__version__} Success!')"

echo "=========================================="
echo "✅ Environment 'mamba_gold' created!"
echo "👉 Run: 'conda activate mamba_gold' to start."
echo "=========================================="
