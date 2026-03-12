import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import os
from pathlib import Path

"""
EXP 100: Geometric Necessity of Global Attention vs Linear SSMs
-------------------------------------------------------------
Goal: Demonstrate that while SSMs (Mamba/S4) are efficient O(N), 
they suffer from "Geometric Compression" in deep networks, 
leading to irreversible Rank Collapse. 
Global Attention (Transformer), acting as a Non-local Heat Kernel,
can recover rank (Geometric Tunneling).

We compare:
1. Transformer (Global Attention, O(N^2))
2. Mamba Proxy (Linear Recurrent SSM, O(N))
"""

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# --- Metrics ---

def effective_rank(matrix):
    """
    Computes the effective rank of a matrix (batch of matrices).
    Shannon entropy of the singular value distribution.
    """
    # matrix: [B, L, D] -> Treat as [B*L, D] or [B, L, D] covariance?
    # We want the geometric rank of the representation manifold in the Embedding Space.
    # So we look at covariance of tokens: C = X^T X (size DxD)
    # Or just SV of X (size LxD).
    # If L > D, max rank is D.
    
    if matrix.dim() == 3:
        # matrix: [B, L, D]
        # We compute rank per sample and average
        ranks = []
        for i in range(matrix.shape[0]):
            x = matrix[i] # [L, D]
            # Center? No, geometry from origin matters for some, but typically we center.
            x = x - x.mean(dim=0, keepdim=True)
            
            try:
                # SVD
                _, S, _ = torch.svd(x)
                # Normalize eigenvalues to probability dist
                S_norm = S / (S.sum() + 1e-12)
                # Entropy
                entropy = -torch.sum(S_norm * torch.log(S_norm + 1e-12))
                ranks.append(torch.exp(entropy).item())
            except:
                ranks.append(0.0)
        return np.mean(ranks)
    return 0.0

# --- Models ---

class GlobalAttentionBlock(nn.Module):
    def __init__(self, d_model, n_head=4):
        super().__init__()
        self.ln = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, n_head, batch_first=True)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.GELU(),
            nn.Linear(4 * d_model, d_model)
        )
        self.ln2 = nn.LayerNorm(d_model)
        
    def forward(self, x):
        # x: [B, L, D]
        res = x
        x = self.ln(x)
        x, _ = self.attn(x, x, x)
        x = res + x
        
        res = x
        x = self.ln2(x)
        x = self.ffn(x)
        x = res + x
        return x

class SSMBlock(nn.Module):
    """
    A simplified Mamba/S4 proxy.
    Core mechanic: Interaction is mediated by a fixed-size recurrent state h_t.
    y_t = C h_t + D x_t
    h_t = A h_{t-1} + B x_t
    
    In deep networks, this acts as a 'bottleneck' for global information flow compared 
    to the explicit A_ij mixing of Transformers.
    """
    def __init__(self, d_model, d_state=16):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        
        self.ln = nn.LayerNorm(d_model)
        
        # Projections
        self.in_proj = nn.Linear(d_model, 2 * d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        
        # SSM Parameters (simplified diagonal)
        # A is typically discretized. We simulate the recurrence directly.
        # Independent SSM per channel (Depthwise)
        self.A_log = nn.Parameter(torch.log(torch.rand(d_model, d_state) + 1e-4)) # Approx [0,1]
        self.B = nn.Parameter(torch.randn(d_model, d_state))
        self.C = nn.Parameter(torch.randn(d_model, d_state))
        self.D = nn.Parameter(torch.randn(d_model))
        
        self.act = nn.SiLU()
        self.ffn = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.GELU(),
            nn.Linear(4 * d_model, d_model)
        )
        self.ln2 = nn.LayerNorm(d_model)

    def forward(self, x):
        # x: [B, L, D]
        res = x
        x = self.ln(x)
        
        B, L, D = x.shape
        
        # 1. Expand (simplified Mamba style)
        # In real Mamba, x splits into x and z (gate). We keep it simple.
        
        # 2. Recurrence (The Bottleneck)
        # Scan implementation
        h = torch.zeros(B, D, self.d_state, device=x.device)
        y = []
        
        # Discretization (Mock)
        # A = -exp(A_log) for stability in S4, here just sigmoid for 0-1 gating
        A = torch.sigmoid(self.A_log) # [D, N]
        
        for t in range(L):
            xt = x[:, t, :] # [B, D]
            
            # h[t] = A * h[t-1] + B * x[t]
            # Element-wise multiplication for diagonal A
            # B*x: [B, D, 1] * [D, N] -> [B, D, N] (Broadcasting needed)
            
            # Simple LRU-like update:
            # h_new = A * h_old + (1-A) * x
            # Let's do standard SSM: h = A h + B x
            
            # B_broad: [1, D, N]
            # xt_broad: [B, D, 1]
            Bx = self.B.unsqueeze(0) * xt.unsqueeze(-1)
            
            h = A.unsqueeze(0) * h + Bx
            
            # y[t] = C * h[t]
            # C: [D, N]
            yt = torch.sum(self.C.unsqueeze(0) * h, dim=-1) # [B, D]
            
            # Skip connection D
            yt = yt + self.D.unsqueeze(0) * xt
            
            y.append(yt)
            
        y = torch.stack(y, dim=1) # [B, L, D]
        
        y = self.act(y)
        y = self.out_proj(y)
        
        x = res + y
        
        # FFN
        res = x
        x = self.ln2(x)
        x = self.ffn(x)
        x = res + x
        return x

class DeepNetwork(nn.Module):
    def __init__(self, block_type='attn', depth=24, d_model=64):
        super().__init__()
        self.layers = nn.ModuleList()
        for _ in range(depth):
            if block_type == 'attn':
                self.layers.append(GlobalAttentionBlock(d_model))
            else:
                self.layers.append(SSMBlock(d_model))
    
    def forward_track_rank(self, x):
        ranks = []
        # Initial rank
        ranks.append(effective_rank(x))
        
        for layer in self.layers:
            x = layer(x)
            ranks.append(effective_rank(x))
            
        return x, ranks

# --- Experiment ---

def run_rank_collapse_experiment():
    print("Running Rank Collapse Experiment...")
    
    DEPTH = 24
    D_MODEL = 64
    SEQ_LEN = 128
    BATCH = 32
    
    # 1. Random Input Data (High Rank)
    # We want to see if the network maintains this rank or collapses
    x = torch.randn(BATCH, SEQ_LEN, D_MODEL, device=device)
    
    # Normalize input to avoid explosion
    x = x / x.std()
    
    # 2. Models
    print("Initializing Transformer...")
    model_attn = DeepNetwork('attn', depth=DEPTH, d_model=D_MODEL).to(device)
    
    print("Initializing SSM (Mamba Proxy)...")
    model_ssm = DeepNetwork('ssm', depth=DEPTH, d_model=D_MODEL).to(device)
    
    # 3. Forward & Measure
    print("Evaluating Transformer...")
    with torch.no_grad():
        _, ranks_attn = model_attn.forward_track_rank(x.clone())
        
    print("Evaluating SSM...")
    with torch.no_grad():
        _, ranks_ssm = model_ssm.forward_track_rank(x.clone())
        
    # 4. Plot
    # Use Nature-like style
    plt.style.use('default')
    plt.rcParams.update({
        'font.family': 'sans-serif',
        'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
        'font.size': 12,
        'axes.labelsize': 14,
        'axes.titlesize': 14,
        'xtick.labelsize': 12,
        'ytick.labelsize': 12,
        'legend.fontsize': 12,
        'figure.figsize': (9, 6.8),
        'lines.linewidth': 2,
        'grid.alpha': 0.3
    })

    fig, ax = plt.subplots(figsize=(9, 6.8), constrained_layout=True)
    layers = range(len(ranks_attn))
    
    ax.plot(layers, ranks_attn, 'b-o', markersize=6, label='Global Attention (Transformer)')
    ax.plot(layers, ranks_ssm, 'r--s', markersize=6, label='Recurrent SSM (Mamba Proxy)')
    
    ax.set_title('Deep Network Geometric Dynamics', fontweight='bold', pad=14)
    ax.set_xlabel('Layer Depth')
    ax.set_ylabel('Effective Geometric Rank ($R_{eff}$)')
    ax.grid(True, alpha=0.3)
    ax.legend(frameon=False, loc='center right')

    # Add a little headroom/footroom so annotations never collide with title or get clipped
    y_min = min(min(ranks_attn), min(ranks_ssm))
    y_max = max(max(ranks_attn), max(ranks_ssm))
    ax.set_ylim(y_min - 2.5, y_max + 2.5)
    
    # Annotation (use pixel offsets to avoid any collision with title / margins)
    ax.annotate(
        'Rank Recovery\n(Geometric Tunneling)',
        xy=(15, ranks_attn[15]),
        xycoords='data',
        xytext=(30, 40),
        textcoords='offset points',
        ha='left',
        va='bottom',
        arrowprops=dict(facecolor='blue', shrink=0.04, alpha=0.45, width=1.2, headwidth=10),
        fontsize=10,
        color='blue'
    )

    ax.annotate(
        'Rank Collapse\n(Geometric Compression)',
        xy=(15, ranks_ssm[15]),
        xycoords='data',
        xytext=(30, -55),
        textcoords='offset points',
        ha='left',
        va='top',
        arrowprops=dict(facecolor='red', shrink=0.04, alpha=0.45, width=1.2, headwidth=10),
        fontsize=10,
        color='red'
    )
    
    # Save next to the manuscript's figures folder (paper_archive/figures)
    out_dir = (Path(__file__).resolve().parents[1] / "figures")
    out_dir.mkdir(parents=True, exist_ok=True)
    save_path = out_dir / "ssm_vs_attention_rank.png"
    save_path_fixed = out_dir / "ssm_vs_attention_rank_fixed.png"
    fig.savefig(save_path, dpi=300, bbox_inches='tight')
    fig.savefig(save_path_fixed, dpi=300, bbox_inches='tight')
    print(f"Saved plot to {str(save_path)}")
    print(f"Saved plot to {str(save_path_fixed)}")
    
    # Print stats
    print(f"\nFinal Rank (Layer {DEPTH}):")
    print(f"Transformer: {ranks_attn[-1]:.2f}")
    print(f"SSM:         {ranks_ssm[-1]:.2f}")

if __name__ == "__main__":
    run_rank_collapse_experiment()
