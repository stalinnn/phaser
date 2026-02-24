import warnings
# Filter warnings immediately before any other imports to catch import-time warnings
warnings.filterwarnings("ignore", category=FutureWarning)

import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from transformers import AutoModelForCausalLM
from mamba_ssm.models.mixer_seq_simple import MambaLMHeadModel
import os

# Set seed
torch.manual_seed(42)

# ==========================================
# 1. Configuration & Models
# ==========================================

# Use A100 optimized settings
DEVICE = "cuda"
DTYPE = torch.bfloat16  # A100 loves bfloat16

# Models to load (Real Scale)
MAMBA_PATH = "/gz-data/falcon-mamba-7b" 
LLAMA_PATH = "/gz-data/Qwen3-8B"

# Hyperparams
SEQ_LEN = 2048 # Long context
BATCH_SIZE = 4
NUM_STEPS = 200 # More steps for convergence

def load_models():
    print(f"Loading Llama from {LLAMA_PATH}...")
    try:
        llama = AutoModelForCausalLM.from_pretrained(
            LLAMA_PATH, 
            torch_dtype=DTYPE,
            device_map="auto",
            trust_remote_code=True
        )
    except Exception as e:
        print(f"Error loading Llama Model: {e}")
        raise e 

    print(f"Loading Mamba from {MAMBA_PATH}...")
    try:
        mamba = AutoModelForCausalLM.from_pretrained(
            MAMBA_PATH,
            torch_dtype=DTYPE,
            device_map="auto",
            trust_remote_code=True
        )
    except Exception:
        print("Retrying Mamba with mamba_ssm...")
        mamba = MambaLMHeadModel.from_pretrained(
            MAMBA_PATH, 
            device=DEVICE, 
            dtype=DTYPE, 
            trust_remote_code=True
        )

    return llama, mamba

# ==========================================
# 2. Connectors (The Core Experiment)
# ==========================================

class LinearConnector(nn.Module):
    """Baseline: Static Linear Map"""
    def __init__(self, dim_in, dim_out):
        super().__init__()
        self.proj = nn.Linear(dim_in, dim_out, bias=False, dtype=DTYPE)
    
    def forward(self, x):
        return self.proj(x)

class LowRankGaugeConnector(nn.Module):
    """NGF: Dynamic Low-Rank Gauge Field"""
    def __init__(self, dim_in, dim_out, rank=32): # Increased rank slightly
        super().__init__()
        self.dim_in = dim_in
        self.dim_out = dim_out
        self.rank = rank
        
        # 1. Base Linear Map (Static)
        self.static_proj = nn.Linear(dim_in, dim_out, bias=False, dtype=DTYPE)
        
        # 2. Dynamic Rotation Generator (Hypernetwork)
        # Generates low-rank factors A_t, B_t for U_t = I + A B^T
        self.hyper_net = nn.Sequential(
            nn.Linear(dim_in, 128, dtype=DTYPE),
            nn.SiLU(), # Use SiLU for better gradient flow
            nn.Linear(128, dim_out * rank * 2, dtype=DTYPE) # Outputs A and B
        )
        
    def forward(self, x):
        # x: [B, L, dim_in]
        
        # 1. Static Projection first
        h_static = self.static_proj(x) # [B, L, dim_out]
        
        # 2. Dynamic Gauge Transformation
        factors = self.hyper_net(x) # [B, L, 2*rank*dim_out]
        A, B_vec = factors.chunk(2, dim=-1)
        
        # Reshape: [B, L, dim_out, rank]
        A = A.view(*x.shape[:2], self.dim_out, self.rank)
        B_vec = B_vec.view(*x.shape[:2], self.dim_out, self.rank)
        
        # Apply U = I + alpha * A B^T
        # alpha is a learned scalar or small constant to keep it perturbative
        # Using einsum: B_vec: b l d r, h_static: b l d -> term1: b l r
        term1 = torch.einsum('bldr,bld->blr', B_vec, h_static)
        # term2: b l d
        gauge_correction = torch.einsum('bldr,blr->bld', A, term1)
        
        # Add normalization to prevent explosion
        return h_static + gauge_correction * 0.01

# ==========================================
# 3. CKA Metric
# ==========================================

def linear_cka(X, Y):
    # X, Y: [L, D]
    X = X.to(torch.float32)
    Y = Y.to(torch.float32)
    
    # Centering
    X = X - X.mean(dim=0, keepdim=True)
    Y = Y - Y.mean(dim=0, keepdim=True)
    
    # Dot product kernel
    gram_x = torch.matmul(X, X.T)
    gram_y = torch.matmul(Y, Y.T)
    
    # CKA
    numerator = torch.trace(torch.matmul(gram_x, gram_y))
    denominator = torch.sqrt(torch.trace(torch.matmul(gram_x, gram_x)) * torch.trace(torch.matmul(gram_y, gram_y)))
    
    return numerator / denominator

# ==========================================
# 4. Main Experiment Loop
# ==========================================

def run_alignment_experiment():
    print("Initializing Experiment on A100 (Mock Input Mode)...")
    
    # 1. Load Models
    llama, mamba = load_models()
    
    # Freeze backbones
    llama.eval()
    mamba.eval()
    for p in llama.parameters(): p.requires_grad = False
    for p in mamba.parameters(): p.requires_grad = False
    
    # Get dimensions
    if hasattr(mamba.config, "d_model"):
        dim_mamba = mamba.config.d_model
    elif hasattr(mamba.config, "hidden_size"):
        dim_mamba = mamba.config.hidden_size
    else:
        dim_mamba = 4096 
        
    dim_llama = llama.config.hidden_size
    print(f"Dimensions: Mamba={dim_mamba}, Llama={dim_llama}")
    
    # 2. Init Connectors
    linear_conn = LinearConnector(dim_mamba, dim_llama).to(DEVICE)
    gauge_conn = LowRankGaugeConnector(dim_mamba, dim_llama).to(DEVICE)
    
    opt_linear = optim.Adam(linear_conn.parameters(), lr=5e-4) # Slightly higher LR
    opt_gauge = optim.Adam(gauge_conn.parameters(), lr=5e-4)
    
    print("Starting Alignment Training...")
    
    cka_history = {"linear": [], "gauge": []}
    
    for step in range(NUM_STEPS):
        # 3. Generate Mock Data (Perfectly Aligned Inputs)
        # Use common vocabulary range (e.g., 0-32000)
        random_input_ids = torch.randint(0, 32000, (BATCH_SIZE, SEQ_LEN)).to(DEVICE)
        
        # Forward pass (Get Targets)
        with torch.no_grad():
            # Mamba Output
            if hasattr(mamba, "backbone"):
                mamba_hidden = mamba.backbone(random_input_ids).last_hidden_state
            else:
                mamba_out_hf = mamba(random_input_ids, output_hidden_states=True)
                mamba_hidden = mamba_out_hf.hidden_states[-1] 
            
            # Llama Target (Middle Layer 16)
            llama_out = llama(random_input_ids, output_hidden_states=True)
            target_hidden = llama_out.hidden_states[16] 
            
        # Ensure inputs match model dtype
        mamba_hidden = mamba_hidden.to(DTYPE)
        target_hidden = target_hidden.to(DTYPE)
            
        # --- Train Linear ---
        opt_linear.zero_grad()
        aligned_linear = linear_conn(mamba_hidden)
        loss_linear = nn.MSELoss()(aligned_linear, target_hidden)
        loss_linear.backward()
        opt_linear.step()
        
        # --- Train Gauge ---
        opt_gauge.zero_grad()
        aligned_gauge = gauge_conn(mamba_hidden)
        loss_gauge = nn.MSELoss()(aligned_gauge, target_hidden)
        loss_gauge.backward()
        opt_gauge.step()
        
        # --- Compute CKA ---
        if step % 10 == 0:
            # Use only first sample to save memory
            cka_l = linear_cka(aligned_linear[0], target_hidden[0]).item()
            cka_g = linear_cka(aligned_gauge[0], target_hidden[0]).item()
            cka_history["linear"].append(cka_l)
            cka_history["gauge"].append(cka_g)
            print(f"Step {step}: Loss L={loss_linear.item():.4f}, G={loss_gauge.item():.4f} | CKA L={cka_l:.3f}, G={cka_g:.3f}")

    # 4. Visualization
    plt.figure(figsize=(10, 6))
    plt.plot(cka_history["linear"], 'o--', label="Linear Adapter (Baseline)")
    plt.plot(cka_history["gauge"], '*-', label="Gauge Connector (Ours)", linewidth=2)
    plt.xlabel("Training Steps (x10)")
    plt.ylabel("CKA Similarity (Mamba -> Llama)")
    plt.title(f"Manifold Alignment (Mock Input): Mamba({dim_mamba}) -> Llama({dim_llama})")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    out_path = "figures/mamba_llama_alignment.png"
    os.makedirs("figures", exist_ok=True)
    plt.savefig(out_path)
    print(f"Saved plot to {out_path}")

if __name__ == "__main__":
    run_alignment_experiment()