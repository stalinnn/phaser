import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from tqdm import tqdm
import os

"""
EXP 50: Curvature-Adaptive Distributed Tensor Dynamics
-------------------------------------------------------
Objective:
Overcome the "Geometric Limits" observed in EXP 41 (Nonlinear Criticality).
We introduce a Curvature-Adaptive Mechanism that modulates the 
trust region (step size) based on local curvature estimation.

Mechanism:
1. Local Estimation of Riemannian Curvature R.
2. If curvature is high (highly nonlinear/unstable region), reduce the effective 
   communication range (damping factor) to prevent "overshooting" on the manifold.
3. This creates a "Trust Region" for the linear approximation.

Physics:
Adaptive damping acts like a "friction" that scales with geometric complexity.
Gamma_eff = Gamma / (1 + beta * |Curvature|)
"""

class AdaptiveNonlinearSystem:
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

    def get_force_and_weights_adaptive(self, x):
        force = np.zeros(self.N)
        force -= (x - self.mu)
        
        diags = np.zeros(self.N) # We will accumulate
        weights = [{} for _ in range(self.N)]
        
        # Track local curvature intensity for adaptation
        # Curvature proxy: Variance of neighbor differences
        local_curvature = np.zeros(self.N)
        
        for i in range(self.N):
            # Calculate local curvature/stress
            diffs = []
            for j in self.adj_list[i]:
                diff = x[i] - x[j]
                diffs.append(np.sin(diff)) # Nonlinear stress
            
            # Simple measure of geometric complexity: Variance of stress forces
            # High variance = Complex/Buckled local geometry
            local_curvature[i] = np.var(diffs) if diffs else 0
            
            # Adaptive Damping Factor (The Core Innovation)
            # If curvature is high, we reduce the effective coupling strength locally
            # This prevents the linear solver from trusting the tangent space too far
            beta = 5.0 # Sensitivity parameter
            adaptive_gamma = self.gamma / (1.0 + beta * local_curvature[i])
            
            diags[i] += 1.0 # From anchor
            
            for j in self.adj_list[i]:
                diff = x[i] - x[j]
                
                # Force uses standard gamma (physics doesn't change)
                force[i] -= self.gamma * np.sin(diff)
                
                # Preconditioner uses ADAPTIVE gamma (solver strategy changes)
                # We trust the connection less if the region is curved
                stiffness = max(0.1, np.cos(diff))
                
                w_val = adaptive_gamma * stiffness
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

def run_adaptive_experiment():
    N = 100
    COUPLING = 20.0 
    STEPS = 1200     
    DT = 0.01
    T = 1.0
    
    # Compare Standard vs Adaptive for increasing K
    k_values = [0, 2, 4, 8, 16, 24, 32]
    
    results_adaptive = []
    
    print(f"Running Curvature-Adaptive Simulation (N={N}, Gamma={COUPLING})...")
    
    sys = AdaptiveNonlinearSystem(n_nodes=N, coupling=COUPLING)
    
    for k in tqdm(k_values):
        # Random start
        x = sys.mu + np.random.uniform(-np.pi, np.pi, N) 
        
        avg_order = 0
        samples = 0
        
        # Transient
        for _ in range(400):
            force, diags, weights = sys.get_force_and_weights_adaptive(x)
            noise = np.random.normal(0, 1, N)
            
            if k == 0:
                dx = force * DT + noise * np.sqrt(2*T*DT)
            else:
                eff_force = sys.solve_distributed(force, diags, weights, k)
                eff_noise = sys.solve_distributed(noise, diags, weights, k//2)
                dx = eff_force * DT + eff_noise * np.sqrt(2*T*DT) * 2.0
            
            x += dx
            
        # Measurement
        for _ in range(STEPS - 400):
            force, diags, weights = sys.get_force_and_weights_adaptive(x)
            noise = np.random.normal(0, 1, N)
            
            if k == 0:
                dx = force * DT + noise * np.sqrt(2*T*DT)
            else:
                eff_force = sys.solve_distributed(force, diags, weights, k)
                eff_noise = sys.solve_distributed(noise, diags, weights, k//2)
                dx = eff_force * DT + eff_noise * np.sqrt(2*T*DT) * 2.0
            
            x += dx
            
            avg_order += sys.get_order_parameter(x)
            samples += 1
            
        results_adaptive.append(avg_order / samples)

    # Print Report
    print("\n--- ADAPTIVE METHOD RESULTS ---")
    print("K\tOrder(R)")
    for idx, k in enumerate(k_values):
        print(f"{k}\t{results_adaptive[idx]:.4f}")

    # Plotting
    os.makedirs('figures', exist_ok=True)
    plt.style.use('bmh')
    
    plt.figure(figsize=(8, 6))
    plt.plot(k_values, results_adaptive, 'o-', color='#e67e22', linewidth=2, markersize=8, label='Curvature Adaptive')
    
    # Add a reference line for "good sync"
    plt.axhline(y=0.8, color='gray', linestyle='--', alpha=0.5, label='Sync Threshold')
    
    plt.title('Overcoming Geometric Limits with Adaptation', fontsize=12)
    plt.xlabel('Negotiation Depth K')
    plt.ylabel('Order Parameter R')
    plt.ylim(0, 1.05)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.savefig('figures/adaptive_success.png', dpi=300)
    print("Saved to figures/adaptive_success.png")

if __name__ == "__main__":
    run_adaptive_experiment()