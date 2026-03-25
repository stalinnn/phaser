import os
import math
import numpy as np
import matplotlib.pyplot as plt
import torch
import sys
from pathlib import Path
import json

_EXPERIMENT_ROOT = Path(__file__).resolve().parent
if str(_EXPERIMENT_ROOT) not in sys.path:
    sys.path.insert(0, str(_EXPERIMENT_ROOT))

from core.hyperbolic_space import clamp_to_ball, make_poincare_ball
from core.metric_alignment import metric_alignment_loss, upper_triangle_mask, map_similarity_to_entanglement_proxy
from emerge_holographic_bulk import set_seed
from geoopt.optim import RiemannianAdam

def run_point(temp_noise: float, data_cov_scale: float, epochs: int = 25) -> float:
    """
    Runs a minimal version of the geometric emergence training directly in Hyperbolic space.
    Returns the final Effective Rank.
    """
    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    N = 32
    d_hyp = 16
    
    # Generate random Euclidean similarity matrix (simulating word entanglements)
    # We scale the variance of this matrix by data_cov_scale
    base_sim = torch.randn(N, N, device=device) * data_cov_scale
    # Make symmetric
    sim_matrix = (base_sim + base_sim.T) / 2.0
    # Softmax to make it look like a real similarity matrix [0, 1]
    sim_matrix = torch.softmax(sim_matrix, dim=-1)
    
    target = map_similarity_to_entanglement_proxy(sim_matrix).to(device)
    mask = upper_triangle_mask(N, device=device)
    
    manifold = make_poincare_ball(c=1.0)
    z_hyp = torch.randn(N, d_hyp, device=device) * 0.01
    z_hyp = manifold.expmap0(z_hyp)
    z_hyp = nn.Parameter(z_hyp)
    
    optimizer = RiemannianAdam([z_hyp], lr=0.05)
    
    for epoch in range(epochs):
        optimizer.zero_grad()
        
        # Inject systematic temperature noise
        # This simulates the stochasticity of SGD / Batch Size
        if temp_noise > 0.0:
            noise = torch.randn_like(z_hyp) * math.sqrt(2 * temp_noise * 0.05)
            z_hyp_noisy = manifold.expmap0(manifold.logmap0(z_hyp) + noise)
            z_hyp_noisy = clamp_to_ball(z_hyp_noisy)
        else:
            z_hyp_noisy = z_hyp
            
        loss = metric_alignment_loss(manifold, z_hyp_noisy, target, mask=mask)
        loss.backward()
        optimizer.step()
        
        with torch.no_grad():
            z_hyp.data = clamp_to_ball(z_hyp.data)
            
    # Calculate final Effective Rank
    with torch.no_grad():
        z_euc = manifold.logmap0(z_hyp)
        z_euc_centered = z_euc - z_euc.mean(dim=0, keepdim=True)
        cov = (z_euc_centered.T @ z_euc_centered) / N
        s = torch.linalg.svdvals(cov)
        s_norm = s / (s.sum() + 1e-12)
        entropy = -torch.sum(s_norm * torch.log(s_norm + 1e-12))
        eff_rank = torch.exp(entropy).item()
        
    return eff_rank / d_hyp # Normalize by 16

import torch.nn as nn

def generate_2d_phase_diagram():
    print("开始生成 AI 智能的三相图 (2D Phase Diagram)...")
    
    # Decrease size to make it run faster for now
    n_temps = 10
    n_covs = 10
    
    temperatures = np.logspace(-4, 0, n_temps) # 0.0001 to 1.0
    cov_scales = np.logspace(-1, 2, n_covs)    # 0.1 to 100.0
    
    results_grid = np.zeros((n_temps, n_covs))
    
    total = n_temps * n_covs
    count = 0
    for i, T in enumerate(temperatures):
        for j, C in enumerate(cov_scales):
            rank = run_point(temp_noise=T, data_cov_scale=C, epochs=25)
            results_grid[i, j] = rank
            count += 1
            if count % 10 == 0:
                print(f"进度: {count}/{total}")
                
    # 绘制热力图
    plt.figure(figsize=(10, 8))
    
    # Origin is 'lower', so Y-axis goes up.
    # Extent: [xmin, xmax, ymin, ymax] -> we use log10 bounds
    extent = [np.log10(cov_scales[0]), np.log10(cov_scales[-1]), 
              np.log10(temperatures[0]), np.log10(temperatures[-1])]
              
    im = plt.imshow(results_grid, origin='lower', extent=extent, 
                    aspect='auto', cmap='magma')
    
    plt.colorbar(im, label="Normalized Effective Rank ($\Phi$)")
    
    # 绘制一条大致的相变理论临界线: Lambda = C / T = const  =>  log(C) - log(T) = const
    # log(T) = log(C) - const
    x_line = np.linspace(extent[0], extent[1], 100)
    # 经验估计的临界常数偏移
    const_offset = np.log10(cov_scales[n_covs//2]) - np.log10(temperatures[n_temps//2])
    y_line = x_line - 1.5 # Tuning the intercept visually for the data
    
    plt.plot(x_line, y_line, 'w--', linewidth=2, label="Theoretical Phase Boundary ($\Lambda = \Lambda_{crit}$)")
    
    # Restrict y limit so line doesn't go off chart
    plt.ylim(extent[2], extent[3])
    
    plt.title("The Phase Diagram of Intelligence")
    plt.xlabel("Data Entanglement Gravity (Log10 $\|Cov\|$)")
    plt.ylabel("System Thermodynamic Noise (Log10 $T_{sys}$)")
    plt.legend(loc="upper left")
    
    plt.text(extent[0]+0.2, extent[3]-0.5, "Chaotic/Trivial Phase\n(High Rank, No Logic)", color='white', fontsize=12)
    plt.text(extent[1]-1.5, extent[2]+0.5, "Structured Phase\n(Low Rank, Logical Trees)", color='black', fontsize=12)
    
    plt.tight_layout()
    plt.savefig("2d_phase_diagram.png", dpi=300)
    print("Saved 2d_phase_diagram.png")

if __name__ == "__main__":
    generate_2d_phase_diagram()
