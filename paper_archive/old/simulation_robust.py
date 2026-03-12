import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from tqdm import tqdm
import os

"""
EXP 61: Robust Nonlinear Distributed Dynamics
-------------------------------------------------------
Addressing the failure of linear extrapolation on curved manifolds.

Mathematical Insight:
The failure at large K is due to "Gradient Explosion" in the tangent space.
On a compact manifold (Torus), the tangent space is only valid locally.
Linear consensus x_j - x_i grows unbounded if not constrained, 
violating the compactness of the manifold.

Solution: Robust Tanh-Diffusion
We introduce a nonlinearity (tanh) in the diffusion step.
This corresponds to a "Bounded Trust Region" in the tangent space.
Physically, this acts as a "soft saturation" of the geometric information flow,
preventing outliers/instabilities from propagating globally.
"""

class RobustSystem:
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

    def get_force_and_metric(self, x):
        force = np.zeros(self.N)
        force -= (x - self.mu)
        
        diags = np.zeros(self.N) 
        weights = [{} for _ in range(self.N)]
        
        for i in range(self.N):
            diags[i] += 1.0 
            
            for j in self.adj_list[i]:
                diff = x[i] - x[j]
                force[i] -= self.gamma * np.sin(diff)
                
                # Use a robust base metric (Saddle-Free-ish but simple)
                # We rely on the solver's nonlinearity for safety
                stiffness = np.abs(np.cos(diff)) + 0.1
                w_val = self.gamma * stiffness
                
                weights[i][j] = w_val
                diags[i] += w_val
                
        return force, diags, weights

    def solve_distributed_nonlinear(self, force, diags, weights, k_steps):
        if k_steps == 0:
            return force 
            
        inv_diag = 1.0 / diags
        u = force * inv_diag # Start with local gradient
        
        # Hyperparameters for Robustness
        alpha = 0.5 
        saturation = 2.0 # Max trust region radius in radians
        
        for _ in range(k_steps):
            Hu = np.zeros(self.N)
            for i in range(self.N):
                s = 0
                for j, w in weights[i].items():
                    # NONLINEAR CORE:
                    # Squash the incoming message.
                    # This prevents a neighbor with a huge 'u' (e.g. crossing a saddle)
                    # from destabilizing this node.
                    # It enforces the "compactness" of the manifold.
                    msg = np.tanh(u[j] / saturation) * saturation
                    s += w * msg
                
                # Self term also saturated? No, self is local.
                Hu[i] = diags[i] * u[i] - s
            
            r = force - Hu
            u = u + alpha * r * inv_diag
            
        return u

def run_robust_experiment():
    N = 100
    COUPLING = 20.0 
    STEPS = 1000     
    DT = 0.01
    T = 1.0 # High temperature to test robustness
    
    k_values = [0, 4, 16, 32, 64]
    results = []
    
    sys = RobustSystem(n_nodes=N, coupling=COUPLING)
    
    print(f"Running Robust Tanh-Dynamics (N={N})...")
    
    for k in tqdm(k_values):
        np.random.seed(42) 
        x = sys.mu + np.random.uniform(-np.pi, np.pi, N) 
        
        avg_order = 0
        
        for t in range(STEPS):
            force, diags, weights = sys.get_force_and_metric(x)
            noise = np.random.normal(0, 1, N)
            
            if k == 0:
                dx = force * DT + noise * np.sqrt(2*T*DT)
            else:
                # Use the Nonlinear Solver
                eff_force = sys.solve_distributed_nonlinear(force, diags, weights, k)
                eff_noise = sys.solve_distributed_nonlinear(noise, diags, weights, k//2)
                dx = eff_force * DT + eff_noise * np.sqrt(2*T*DT) * 2.0
            
            x += dx
            
            if t > 500:
                avg_order += sys.get_order_parameter(x)
        
        results.append(avg_order / 500)

    print("\n--- ROBUST RESULTS ---")
    print("K\tOrder(R)")
    for i, k in enumerate(k_values):
        print(f"{k}\t{results[i]:.4f}")

    # Plot
    os.makedirs('figures', exist_ok=True)
    plt.style.use('bmh')
    plt.figure(figsize=(8, 6))
    plt.plot(k_values, results, 'o-', color='#8e44ad', linewidth=3, label='Robust Tanh-Dynamics')
    plt.axhline(y=0.8, color='gray', linestyle='--')
    plt.title('Robust Coordination on Nonlinear Manifolds', fontsize=12)
    plt.xlabel('Negotiation Depth K')
    plt.ylabel('Order Parameter R')
    plt.savefig('figures/robust_nonlinear.png', dpi=300)

if __name__ == "__main__":
    run_robust_experiment()
