import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import os
import time

"""
EXP 90: 3D Spin Glass Optimization (Edwards-Anderson Model)
---------------------------------------------------------
Physics Goal: 
Demonstrate that Attention breaks the "Exponential Time Barrier" in finding 
ground states of frustrated systems (NP-Hard).

Comparison:
1. Langevin Dynamics (Local)
2. Random Long-Range Links (Small World Baseline)
3. Attention Dynamics (Geometric Tunneling)
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
        # Shape [L, L, L]
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
            
            # Gradient Descent
            d_p = p.grad.data
            
            # Add Thermal Noise (Langevin)
            # dx = -grad * dt + sqrt(2*T*dt) * noise
            noise = torch.randn_like(p.data) * np.sqrt(2 * self.T * self.lr)
            
            p.data.add_(-self.lr * d_p + noise)

class RandomLinkOptimizer:
    """
    Baseline: Adds random long-range couplings to simulate Small World networks.
    Each node receives gradients from k random neighbors.
    """
    def __init__(self, params, L, lr=0.05, temperature=0.05, k_neighbors=4):
        self.params = list(params)
        self.lr = lr
        self.T = temperature
        self.k = k_neighbors
        self.L = L
        self.N = L*L*L
        
    def step(self):
        for p in self.params:
            if p.grad is None: continue
            
            # 1. Local Gradient
            g_local = p.grad.data
            
            # 2. Random Long-Range Gradient Aggregation
            # Simulates fixed or dynamic random links
            # We use dynamic random links here (mean field approximation)
            g_flat = g_local.view(g_local.shape[0], -1, 1)
            
            g_rand_sum = torch.zeros_like(g_flat)
            for _ in range(self.k):
                perm = torch.randperm(self.N, device=p.device)
                g_rand_sum += g_flat[:, perm, :]
                
            g_random = (g_rand_sum / self.k).view_as(g_local)
            
            # Mix: 30% Local, 70% Random (Strong Coupling)
            g_effective = 0.3 * g_local + 0.7 * g_random
            
            # Update
            noise = torch.randn_like(p.data) * np.sqrt(2 * self.T * self.lr)
            p.data.add_(-self.lr * g_effective + noise)

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
        self.N = L*L*L
        
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
            
            grads = p.grad.data
            spins = p.data
            
            BATCH = spins.shape[0]
            
            # Flatten
            g_flat = grads.view(BATCH, -1, 1) # V
            s_flat = spins.view(BATCH, -1, 1)
            p_flat = self.pos_flat.expand(BATCH, -1, -1)
            
            # --- Attention Mechanism ---
            # Smart Logic: "Frustration Attention"
            # Query: My Gradient Norm (How frustrated am I?)
            # Key:   Your Gradient Norm (How stable are you?)
            # We want High Grad nodes to attend to Low Grad nodes.
            
            g_norm = torch.norm(g_flat, dim=2, keepdim=True) # [B, N, 1]
            
            # Distance in State Space (to ensure we don't jump to totally unrelated basins)
            d_state = torch.cdist(s_flat, s_flat)
            
            # Frustration Difference:
            # Score ~ (My_Grad - Your_Grad)
            # If I have High Grad (10) and you have Low (1), Score is High.
            # If I have Low (1) and you have High (10), Score is Low.
            
            # Q = g_norm
            # K = -g_norm
            # Score = Q + K.T = g_i - g_j
            # We use Softmax, so large positive means "Attract".
            
            # Simple Attention Score:
            # A_ij ~ exp( beta * ( |g_i| - |g_j| ) - distance_penalty )
            # Meaning: "I pull information from those more stable than me."
            
            frustration_diff = g_norm - g_norm.transpose(1, 2) # [B, N, N]
            
            # Geometric constraints (soft locality)
            d_pos = torch.cdist(p_flat, p_flat)
            
            # SCALING IS CRITICAL FOR L=8
            # Distances can be large (up to 8*sqrt(3) ~ 14). d^2 ~ 200.
            # Frustration diff ~ 1-10.
            # We need to balance them.
            
            beta = 2.0
            # Normalize distance by system size L
            dist_term = (d_pos / self.L) ** 2 
            
            scores = beta * frustration_diff - 5.0 * dist_term
            
            weights = torch.softmax(scores, dim=-1)
            
            # Smooth Gradient (The "Guide" field)
            # We import the gradients from the stable neighbors
            g_smooth = torch.bmm(weights, g_flat).view_as(grads)
            
            # Strong injection
            g_effective = grads + 10.0 * g_smooth
            
            # Update
            noise = torch.randn_like(spins) * np.sqrt(2 * self.T * self.lr)
            p.data.add_(-self.lr * g_effective + noise)

# --- Experiment Runner ---

def run_spin_glass_experiment():
    print("\n>>> STARTING 3D SPIN GLASS EXPERIMENT (EDWARDS-ANDERSON) <<<")
    
    L = 8 
    BATCH_SIZE = 8 
    STEPS = 800
    
    system = SpinGlass3D(L=L)
    
    # 1. Langevin
    print("Running Langevin Dynamics...")
    spins_l = torch.randn(BATCH_SIZE, L, L, L, device=device, requires_grad=True)
    opt_l = LangevinOptimizer([spins_l], lr=0.05)
    hist_l = []
    min_l = float('inf')
    
    for t in tqdm(range(STEPS)):
        opt_l.params[0].grad = None
        loss = system.energy(spins_l).mean()
        loss.backward()
        opt_l.step()
        min_l = min(min_l, system.energy(spins_l).min().item())
        hist_l.append(min_l)
        
    # 2. Random Link
    print("Running Random Link Dynamics...")
    spins_r = torch.randn(BATCH_SIZE, L, L, L, device=device, requires_grad=True)
    opt_r = RandomLinkOptimizer([spins_r], L=L, lr=0.05)
    hist_r = []
    min_r = float('inf')
    
    for t in tqdm(range(STEPS)):
        opt_r.params[0].grad = None
        loss = system.energy(spins_r).mean()
        loss.backward()
        opt_r.step()
        min_r = min(min_r, system.energy(spins_r).min().item())
        hist_r.append(min_r)

    # 3. Attention
    print("Running Attention Dynamics...")
    spins_a = torch.randn(BATCH_SIZE, L, L, L, device=device, requires_grad=True)
    opt_a = AttentionOptimizer([spins_a], L=L, lr=0.05)
    hist_a = []
    min_a = float('inf')
    
    for t in tqdm(range(STEPS)):
        opt_a.params[0].grad = None
        loss = system.energy(spins_a).mean()
        loss.backward()
        opt_a.step()
        min_a = min(min_a, system.energy(spins_a).min().item())
        hist_a.append(min_a)
        
    print(f"\nFinal Energies:")
    print(f"Langevin: {min_l:.2f}")
    print(f"Random:   {min_r:.2f}")
    print(f"Attention:{min_a:.2f}")
    
    # Plot
    os.makedirs('figures', exist_ok=True)
    plt.figure(figsize=(10, 6))
    plt.plot(hist_l, 'r-', linewidth=1.5, alpha=0.8, label='Langevin (Local)')
    plt.plot(hist_r, 'b--', linewidth=1.5, alpha=0.8, label='Random Long-Range (Small World)')
    plt.plot(hist_a, 'g-', linewidth=2.5, label='Attention (Geometric Tunneling)')
    plt.title(f'3D Spin Glass Optimization (L={L}, N={L**3})\n"Smart" vs "Blind" Connectivity', fontsize=14)
    plt.xlabel('Optimization Steps', fontsize=12)
    plt.ylabel('System Energy (Lower is Better)', fontsize=12)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig('figures/spin_glass_3d_proof.png', dpi=300)
    print("Saved to figures/spin_glass_3d_proof.png")

if __name__ == "__main__":
    run_spin_glass_experiment()
