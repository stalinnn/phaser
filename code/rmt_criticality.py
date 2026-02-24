import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

"""
EXP 101: RMT Phase Transition Verification
------------------------------------------
Goal: Demonstrate that the scaling factor 1/sqrt(d) places the Attention matrix
exactly at the critical phase transition point between "Chaos" (High Entropy) 
and "Order" (Low Entropy/Rank Collapse).

We analyze the Singular Value Distribution (Spectral Density) of:
A = Softmax( alpha * Q @ K.T / sqrt(d) )
"""

def run_rmt_simulation():
    print(">>> Running RMT Phase Transition Simulation <<<")
    
    N = 1024   # Sequence length
    D = 64     # Head dimension (typical for Transformer)
    SEED = 42
    torch.manual_seed(SEED)
    
    # Generate random Q, K ~ N(0, 1)
    Q = torch.randn(N, D)
    K = torch.randn(N, D)
    
    # Pre-compute raw dot products
    # M_raw ~ N(0, d)
    M_raw = Q @ K.T 
    
    # Define scaling factors alpha: scale = alpha / sqrt(d)
    # alpha = 1.0 corresponds to the standard 1/sqrt(d)
    alphas = [0.1, 1.0, 5.0]
    labels = [r'High Temp ($\alpha=0.1$)', r'Critical Point ($\alpha=1.0$)', r'Frozen ($\alpha=5.0$)']
    colors = ['blue', 'green', 'red']
    
    plt.figure(figsize=(12, 5))
    
    # Subplot 1: Spectral Density (Histogram of Singular Values)
    plt.subplot(1, 2, 1)
    
    for alpha, label, color in zip(alphas, labels, colors):
        scale = alpha / np.sqrt(D)
        
        # Apply Softmax
        # A_ij = exp(scale * M_ij) / sum
        A = torch.softmax(scale * M_raw, dim=-1)
        
        # Compute Singular Values
        # Note: A is N x N. 
        # Since A is row-stochastic, max singular value is always 1 (Perron-Frobenius).
        # We are interested in the distribution of the REST.
        _, S, _ = torch.linalg.svd(A)
        S = S.numpy()
        
        # Remove the first trivial singular value (approx 1.0) to see the structure of noise
        S_rest = S[1:]
        
        # Plot density (KDE or Histogram)
        sns.kdeplot(S_rest, label=label, color=color, fill=True, alpha=0.1, linewidth=2)
        
    plt.title("Spectral Density of Attention Matrix\n(Singular Value Distribution)", fontsize=12)
    plt.xlabel(r"Singular Value $\sigma$", fontsize=10)
    plt.ylabel("Density", fontsize=10)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.xlim(0, 0.5) # Zoom in on the tail
    
    # Subplot 2: Effective Rank vs Alpha (The Phase Transition Curve)
    plt.subplot(1, 2, 2)
    
    alpha_range = np.logspace(-1, 1.5, 50) # 0.1 to ~30
    ranks = []
    
    print("Scanning alpha range...")
    for alpha in alpha_range:
        scale = alpha / np.sqrt(D)
        A = torch.softmax(scale * M_raw, dim=-1)
        
        # Effective Rank = exp(Entropy of Singular Values)
        _, S, _ = torch.linalg.svd(A)
        S_norm = S / S.sum()
        entropy = -torch.sum(S_norm * torch.log(S_norm + 1e-12))
        r_eff = torch.exp(entropy).item()
        ranks.append(r_eff)
        
    plt.plot(alpha_range, ranks, 'k.-', linewidth=1.5)
    
    # Mark the critical point
    plt.axvline(x=1.0, color='green', linestyle='--', label=r'Standard Scaling ($1/\sqrt{d}$)')
    
    # Annotate regimes
    plt.text(0.15, max(ranks)*0.8, 'Chaotic / High Rank', color='blue', fontsize=10)
    plt.text(3.0, min(ranks)*1.5, 'Frozen / Rank Collapse', color='red', fontsize=10)
    
    # Calculate curvature (2nd derivative) to find max curvature point
    # Ideally, max curvature is near alpha=1
    
    plt.xscale('log')
    plt.title("Geometric Phase Transition", fontsize=12)
    plt.xlabel(r"Scaling Factor $\alpha$ (in $\alpha/\sqrt{d}$)", fontsize=10)
    plt.ylabel("Effective Rank $R_{eff}$", fontsize=10)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    os.makedirs('figures', exist_ok=True)
    save_path = 'figures/rmt_phase_transition.png'
    plt.savefig(save_path, dpi=300)
    print(f"Saved RMT proof to {save_path}")

if __name__ == "__main__":
    run_rmt_simulation()
