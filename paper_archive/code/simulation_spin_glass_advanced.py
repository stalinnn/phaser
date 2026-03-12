import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import os
import time

"""
EXP 95: 3D Spin Glass Optimization - The Ultimate Benchmark
---------------------------------------------------------
Comparing:
1. Langevin Dynamics (Baseline)
2. Parallel Tempering (Physics SOTA)
3. Attention Dynamics (Ours)

Goal: Prove Attention is competitive with or superior to advanced MCMC methods
for finding ground states in frustrated systems.
"""

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

class SpinGlass3D:
    def __init__(self, L=8, seed=42):
        self.L = L
        self.N = L * L * L
        torch.manual_seed(seed)
        np.random.seed(seed)
        
        # Random couplings J ~ N(0, 1)
        self.Jx = torch.randn(L, L, L, device=device)
        self.Jy = torch.randn(L, L, L, device=device)
        self.Jz = torch.randn(L, L, L, device=device)
        
    def energy(self, spins):
        # spins: [Batch, L, L, L]
        s_right = torch.roll(spins, shifts=-1, dims=1)
        s_down  = torch.roll(spins, shifts=-1, dims=2)
        s_back  = torch.roll(spins, shifts=-1, dims=3)
        
        E_x = -torch.sum(self.Jx * spins * s_right, dim=(1,2,3))
        E_y = -torch.sum(self.Jy * spins * s_down,  dim=(1,2,3))
        E_z = -torch.sum(self.Jz * spins * s_back,  dim=(1,2,3))
        
        # Soft constraint to keep spins near +/- 1
        E_well = torch.sum((spins**2 - 1)**2, dim=(1,2,3))
        
        return E_x + E_y + E_z + 1.0 * E_well

# --- Optimizers ---

class LangevinOptimizer:
    def __init__(self, params, lr=0.05, temperature=0.05):
        self.params = list(params)
        self.lr = lr
        self.T = temperature
        
    def step(self):
        for p in self.params:
            if p.grad is None: continue
            d_p = p.grad.data
            # Clip gradients to prevent explosion
            torch.nn.utils.clip_grad_norm_(p, 1.0)
            d_p = p.grad.data # Reload after clipping if in-place, but clip_grad_norm_ works on p.grad
            
            noise = torch.randn_like(p.data) * np.sqrt(2 * self.T * self.lr)
            p.data.add_(-self.lr * d_p + noise)

class ParallelTemperingOptimizer:
    """
    Simulates multiple replicas at different temperatures and allows them to swap.
    This helps the system escape local minima (high T) and settle into ground states (low T).
    """
    def __init__(self, params, system, n_replicas=8, T_min=0.1, T_max=5.0, lr=0.05):
        self.spins = params[0] # Expecting [n_replicas, L, L, L]
        self.system = system
        self.lr = lr
        self.n_replicas = n_replicas
        
        # Geometric spacing for temperatures
        self.temps = torch.logspace(np.log10(T_min), np.log10(T_max), n_replicas).to(device)
        self.betas = 1.0 / self.temps
        
    def step(self):
        # 1. Langevin Step for each replica (vectorized)
        if self.spins.grad is not None:
            # Clip gradients
            torch.nn.utils.clip_grad_norm_([self.spins], 1.0)
            
            d_p = self.spins.grad.data
            # Noise depends on T
            noise_scale = torch.sqrt(2 * self.temps * self.lr).view(-1, 1, 1, 1)
            noise = torch.randn_like(self.spins) * noise_scale
            self.spins.data.add_(-self.lr * d_p + noise)
            
        # 2. Replica Exchange (Swap)
        # We attempt to swap i and i+1
        with torch.no_grad():
            E = self.system.energy(self.spins) # [n_replicas]
            
            # Iterate through pairs
            for i in range(self.n_replicas - 1):
                j = i + 1
                
                # Metropolis Criterion for swapping CONFIGURATIONS
                # P = min(1, exp( (beta_i - beta_j) * (E_i - E_j) ))
                d_beta = self.betas[i] - self.betas[j] # Positive since T_i < T_j
                d_E = E[i] - E[j]
                
                delta = d_beta * d_E
                
                if delta > 0 or torch.rand(1, device=device).item() < torch.exp(delta).item():
                    # Swap configurations
                    temp = self.spins.data[i].clone()
                    self.spins.data[i] = self.spins.data[j]
                    self.spins.data[j] = temp
                    
                    # CRITICAL FIX: Update Energies after swap so next pair uses correct values
                    temp_E = E[i].clone()
                    E[i] = E[j]
                    E[j] = temp_E

class AttentionOptimizer(nn.Module):
    """
    Ours: Uses Geometric Attention to smooth gradients.
    """
    def __init__(self, params, L, lr=0.05, temperature=0.05):
        super().__init__()
        self.params = list(params)
        self.lr = lr
        self.T = temperature
        self.L = L
        
        # Positional Encoding
        grid_x, grid_y, grid_z = torch.meshgrid(
            torch.linspace(-1, 1, L),
            torch.linspace(-1, 1, L),
            torch.linspace(-1, 1, L),
            indexing='ij'
        )
        self.pos = torch.stack([grid_x, grid_y, grid_z], dim=-1).to(device) # [L,L,L,3]
        self.pos_flat = self.pos.reshape(1, -1, 3) # [1, N, 3]
        
    def step(self):
        for p in self.params:
            if p.grad is None: continue
            
            # Clip gradients
            torch.nn.utils.clip_grad_norm_(p, 1.0)
            
            grads = p.grad.data
            spins = p.data
            BATCH = spins.shape[0]
            
            # Flatten
            g_flat = grads.view(BATCH, -1, 1) 
            s_flat = spins.view(BATCH, -1, 1)
            p_flat = self.pos_flat.expand(BATCH, -1, -1)
            
            g_norm = torch.norm(g_flat, dim=2, keepdim=True) # [B, N, 1]
            
            # Attention Mechanism
            frustration_diff = g_norm - g_norm.transpose(1, 2) 
            d_pos = torch.cdist(p_flat, p_flat)
            
            beta = 50.0
            dist_term = 0.0 # Allow global tunneling
            scores = beta * frustration_diff - 0.0 * dist_term
            
            weights = torch.softmax(scores, dim=-1)
            
            # Value = Spins (State)
            # We pull towards the states of stable (low frustration) neighbors
            # This implements "Geometric Smoothing" of the state field
            v_flat = s_flat
            
            # Smooth State
            s_smooth = torch.bmm(weights, v_flat).view_as(spins)
            
            # Geometric Force: Pull towards the smooth consensus
            # F = k * (s_smooth - s_current)
            f_geo = s_smooth - spins
            
            # Update
            # We add this geometric force to the negative gradient
            g_effective = grads - 50.0 * f_geo
            
            noise = torch.randn_like(spins) * np.sqrt(2 * self.T * self.lr)
            p.data.add_(-self.lr * g_effective + noise)

# --- Experiment Runner ---

def run_benchmark():
    print("\n>>> STARTING SPIN GLASS BENCHMARK: ATTENTION vs PHYSICS SOTA <<<")
    
    L = 6 # Slightly smaller for speed/memory if we run many replicas
    BATCH_SIZE = 8 # Number of parallel runs (chains) per method
    # Note: For Parallel Tempering, BATCH_SIZE will be number of replicas in ONE system.
    # To be fair, we should run multiple independent PT systems or just compare the best of the replicas.
    # Let's run 1 PT system with 8 replicas, and compare to 8 independent Langevin chains, and 8 Attention chains.
    
    STEPS = 2000
    
    system = SpinGlass3D(L=L)
    
    # 1. Langevin (Baseline) - 8 independent walkers at low T
    print("Running Langevin Dynamics...")
    spins_l = torch.randn(BATCH_SIZE, L, L, L, device=device, requires_grad=True)
    opt_l = LangevinOptimizer([spins_l], lr=0.01, temperature=0.1)
    hist_l = []
    best_l = float('inf')
    
    for t in tqdm(range(STEPS)):
        opt_l.params[0].grad = None
        loss = system.energy(spins_l).sum() # Sum so we get gradients for all
        loss.backward()
        opt_l.step()
        # Track min energy across the batch
        curr_min = system.energy(spins_l).min().item()
        best_l = min(best_l, curr_min)
        hist_l.append(best_l)
        
    # 2. Parallel Tempering (Physics SOTA) - 8 replicas at T=[0.1 ... 1.5]
    # We track the energy of the lowest temperature replica (index 0) or the min across all (since we want the ground state)
    print("Running Parallel Tempering...")
    spins_pt = torch.randn(BATCH_SIZE, L, L, L, device=device, requires_grad=True)
    opt_pt = ParallelTemperingOptimizer([spins_pt], system, n_replicas=BATCH_SIZE, T_min=0.1, T_max=1.5, lr=0.01)
    hist_pt = []
    best_pt = float('inf')
    
    for t in tqdm(range(STEPS)):
        opt_pt.spins.grad = None
        loss = system.energy(spins_pt).sum()
        loss.backward()
        opt_pt.step()
        
        # Track min energy found by ANY replica
        curr_min = system.energy(spins_pt).min().item()
        best_pt = min(best_pt, curr_min)
        hist_pt.append(best_pt)

    # 3. Attention Dynamics (Ours) - 8 independent walkers at low T
    print("Running Attention Dynamics...")
    spins_a = torch.randn(BATCH_SIZE, L, L, L, device=device, requires_grad=True)
    opt_a = AttentionOptimizer([spins_a], L=L, lr=0.01, temperature=0.1)
    hist_a = []
    best_a = float('inf')
    
    for t in tqdm(range(STEPS)):
        opt_a.params[0].grad = None
        loss = system.energy(spins_a).sum()
        loss.backward()
        opt_a.step()
        curr_min = system.energy(spins_a).min().item()
        best_a = min(best_a, curr_min)
        hist_a.append(best_a)
        
    print(f"\nFinal Min Energies:")
    print(f"Langevin (Local): {hist_l[-1]:.2f}")
    print(f"Parallel Tempering (Standard): {hist_pt[-1]:.2f}")
    print(f"Attention (Geometric): {hist_a[-1]:.2f}")
    
    # Plot
    os.makedirs('figures', exist_ok=True)
    plt.figure(figsize=(10, 6))
    plt.plot(hist_l, 'r-', linewidth=1.5, alpha=0.6, label='Langevin (Local Gradient)')
    plt.plot(hist_pt, 'b--', linewidth=2.0, alpha=0.8, label='Parallel Tempering (Thermodynamic Baseline)')
    plt.plot(hist_a, 'g-', linewidth=2.5, label='Attention Dynamics (Geometric Tunneling)')
    
    plt.title(f'3D Spin Glass Optimization (L={L}, N={L**3})\nComplexity Class Comparison', fontsize=14)
    plt.xlabel('Optimization Steps', fontsize=12)
    plt.ylabel('System Energy (Lower is Better)', fontsize=12)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig('figures/spin_glass_sota_comparison.png', dpi=300)
    print("Saved to figures/spin_glass_sota_comparison.png")

if __name__ == "__main__":
    run_benchmark()
