# Thermodynamic Gated Networks (TGN)

Official Implementation for the Nature Machine Intelligence submission:
**"Thermodynamic Gated Networks (TGN): Attention as a Geometric Anti-Dissipative Force"**

## 1. Environment Setup

### Requirements
- Python 3.10+
- PyTorch 2.0+ (CUDA 11.8/12.1 Recommended)
- `mamba_ssm` (Optional, for SOTA comparison)

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install tiktoken datasets tqdm matplotlib pandas causal-conv1d mamba-ssm
```

## 2. Repository Structure

```
TGN-Nature/
├── tgn/
│   └── modeling.py       # Core TGN Architecture (GRU+Attention+Gate)
├── experiments/
│   ├── lm_convergence.py # Fig 21: WikiText-103 Convergence
│   ├── sota_battle.py    # Fig 22: Early Dynamics vs Mamba
│   └── mqar.py           # Fig 22a: MQAR Capacity Test
├── scripts/
│   └── run_wikitext.sh   # Bash script for cloud training
└── figures/              # Generated plots
```

## 3. Reproduction Instructions

### Experiment 1: WikiText-103 Convergence (Fig 21)
This experiment trains a 125M parameter TGN model on WikiText-103 to demonstrate the "Hysteresis Awakening" of the gate mechanism.

```bash
# Run on single GPU or CPU (Demo)
python experiments/lm_convergence.py

# Run on Cloud (6x A100 DDP)
bash scripts/run_wikitext.sh
```

### Experiment 2: SOTA Comparison (Fig 22)
Compare TGN against Mamba and Transformer on early training dynamics.

```bash
# Run SOTA battle (requires 6 GPUs)
torchrun --nproc_per_node=6 experiments/sota_battle.py --model_type tgn --model_size medium
```

### Experiment 3: MQAR Capacity (Fig 22a)
Test the memory capacity limit of TGN vs Mamba.

```bash
python experiments/mqar.py
```

## 4. Citation
If you find this work useful, please cite our Nature Machine Intelligence paper.
