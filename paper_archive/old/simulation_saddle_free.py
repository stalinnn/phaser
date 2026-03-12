import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from tqdm import tqdm
import os

"""
EXP 60: Saddle-Free Distributed Dynamics
-------------------------------------------------------
Addressing the Mathematical Flaw in Section 3.4
The previous "Adaptive" mechanism failed at large K because it treated 
negative curvature (instability) as "flatness" (clipping cos(theta) to 0.1).
This caused the solver to take excessively large steps in unstable regions.

Solution: Saddle-Free Newton Dynamics
We construct the metric G using the Absolute Value of the curvature:
G_ij ~ |cos(theta_i - theta_j)|
This ensures that high curvature (whether stable valley or unstable peak)
results in cautious (damped) steps, preventing overshooting.
"""

class SaddleFreeSystem:
    def __init__(self, n_nodes=100, k_neighbors=4, coupling=20.0, seed=42):
        self.N = n_nodes
        self.gamma = coupling
        np.random.seed(seed)
        
        # Topology: Small World Ring
        self.G = nx.watts_strogatz_graph(n_nodes, k=k_neighbors, p=0.1, seed=seed)
        self.adj_list = [list(self.G.neighbors(i)) for i in range(n_nodes)]
        
        # Target: Anchored Kuramoto
        xs = np.linspace(0, 2*np.pi, n_nodes)
        self.mu = xs 
        
    def get_order_parameter(self, x):
        deviations = x - self.mu
        z = np.mean(np.exp(1j * deviations))
        return np.abs(z)

    def get_force_and_metric(self, x, method='clipped'):
        """
        method: 
          'clipped' -> The original flawed approach (max(0.1, cos))
          'saddle_free' -> The mathematically correct approach (|cos|)
        """
        force = np.zeros(self.N)
        force -= (x - self.mu)
        
        diags = np.zeros(self.N) 
        weights = [{} for _ in range(self.N)]
        
        for i in range(self.N):
            diags[i] += 1.0 # Anchor
            
            for j in self.adj_list[i]:
                diff = x[i] - x[j]
                
                # 1. Force (Gradient) - Always the same physics
                force[i] -= self.gamma * np.sin(diff)
                
                # 2. Metric (Preconditioner) - The Geometry of Information
                cos_d = np.cos(diff)
                
                if method == 'clipped':
                    # ORIGINAL "BUGGY" LOGIC
                    # If cos_d < 0 (unstable), we treat it as small (0.1).
                    # Small metric = Large step -> Overshoot!
                    stiffness = max(0.1, cos_d)
                elif method == 'saddle_free':
                    # SADDLE-FREE LOGIC
                    # If cos_d < 0, we treat it as high curvature (|cos|).
                    # Large metric = Small step -> Stability!
                    stiffness = np.abs(cos_d) + 0.1 # epsilon for safety
                
                # We use the standard gamma, no "adaptive variance" needed if math is right
                w_val = self.gamma * stiffness
                
                weights[i][j] = w_val
                diags[i] += w_val
                
        return force, diags, weights

    def solve_distributed(self, force, diags, weights, k_steps):
        if k_steps == 0:
            return force 
            
        inv_diag = 1.0 / diags
        u = force * inv_diag
        alpha = 0.5 
        
        for _ in range(k_steps):
            Hu = np.zeros(self.N)
            for i in range(self.N):
                s = 0
                for j, w in weights[i].items():
                    s += w * u[j]
                Hu[i] = diags[i] * u[i] - s
            
            r = force - Hu
            u = u + alpha * r * inv_diag
            
        return u

def run_comparison():
    N = 100
    COUPLING = 20.0 
    STEPS = 1000     
    DT = 0.01
    T = 1.0
    
    # We test large K where the original failed
    k_values = [0, 4, 16, 32, 64]
    
    modes = ['clipped', 'saddle_free']
    results = {m: [] for m in modes}
    
    sys = SaddleFreeSystem(n_nodes=N, coupling=COUPLING)
    
    print(f"Comparing Clipped vs Saddle-Free Dynamics (N={N})...")
    
    for mode in modes:
        print(f"Testing Mode: {mode}")
        for k in tqdm(k_values):
            np.random.seed(42) # Same start for fair comparison
            x = sys.mu + np.random.uniform(-np.pi, np.pi, N) 
            
            avg_order = 0
            
            # Run simulation
            for t in range(STEPS):
                force, diags, weights = sys.get_force_and_metric(x, method=mode)
                noise = np.random.normal(0, 1, N)
                
                if k == 0:
                    dx = force * DT + noise * np.sqrt(2*T*DT)
                else:
                    eff_force = sys.solve_distributed(force, diags, weights, k)
                    eff_noise = sys.solve_distributed(noise, diags, weights, k//2)
                    dx = eff_force * DT + eff_noise * np.sqrt(2*T*DT) * 2.0
                
                x += dx
                
                if t > 500: # Steady state
                    avg_order += sys.get_order_parameter(x)
            
            results[mode].append(avg_order / 500)

    # Plotting
    os.makedirs('figures', exist_ok=True)
    plt.style.use('bmh')
    plt.figure(figsize=(8, 6))
    
    plt.plot(k_values, results['clipped'], 'o--', color='gray', label='Original (Clipped)')
    plt.plot(k_values, results['saddle_free'], 'o-', color='#27ae60', linewidth=3, label='Saddle-Free (Corrected)')
    
    plt.title('Resolution of Section 3.4: Saddle-Free Dynamics', fontsize=12)
    plt.xlabel('Negotiation Depth K')
    plt.ylabel('Order Parameter R')
    plt.ylim(0, 1.05)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.savefig('figures/saddle_free_fix.png', dpi=300)
    print("Saved to figures/saddle_free_fix.png")
    
    # Print numerical comparison
    print("\n--- RESULTS SUMMARY ---")
    print("K\tOriginal\tSaddle-Free")
    for i, k in enumerate(k_values):
        print(f"{k}\t{results['clipped'][i]:.4f}\t\t{results['saddle_free'][i]:.4f}")

if __name__ == "__main__":
    run_comparison()
