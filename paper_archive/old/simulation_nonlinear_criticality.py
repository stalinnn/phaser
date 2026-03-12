import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from tqdm import tqdm
import os

"""
EXP 41: Criticality Scan on Nonlinear Manifolds (Kuramoto Model)
----------------------------------------------------------------
Objective: 
Determine if the "Critical Communication Depth" (K_c) shifts when 
the underlying geometry becomes non-Euclidean (state-dependent curvature).

Hypothesis:
Nonlinearity introduces local potential wells and barriers. 
The system might require a higher K_c to average out the local curvature 
distortions compared to the linear case (where K_c was approx 2).
"""

class NonlinearCriticalitySystem:
    def __init__(self, n_nodes=100, k_neighbors=4, coupling=8.0, seed=42):
        self.N = n_nodes
        self.gamma = coupling
        np.random.seed(seed)
        
        # Topology: Small World Ring
        self.G = nx.watts_strogatz_graph(n_nodes, k=k_neighbors, p=0.1, seed=seed)
        self.adj_list = [list(self.G.neighbors(i)) for i in range(n_nodes)]
        
        # Target: Anchored Kuramoto
        xs = np.linspace(0, 2*np.pi, n_nodes)
        self.mu = xs # Target phase pattern
        
    def get_order_parameter(self, x):
        # Order Parameter R relative to target
        deviations = x - self.mu
        z = np.mean(np.exp(1j * deviations))
        return np.abs(z)

    def get_energy(self, x):
        e_anchor = 0.5 * np.sum((x - self.mu)**2)
        e_inter = 0
        for i in range(self.N):
            for j in self.adj_list[i]:
                if i < j:
                    e_inter += self.gamma * (1 - np.cos(x[i] - x[j]))
        return e_anchor + e_inter

    def get_force_and_weights(self, x):
        force = np.zeros(self.N)
        force -= (x - self.mu)
        
        diags = np.ones(self.N)
        weights = [{} for _ in range(self.N)]
        
        for i in range(self.N):
            for j in self.adj_list[i]:
                diff = x[i] - x[j]
                
                # Nonlinear Force: sin(diff)
                force[i] -= self.gamma * np.sin(diff)
                
                # Nonlinear Hessian Proxy: cos(diff)
                # Key Difference: Stiffness can be negative! (Repulsive/Unstable zone)
                # We clip it to be positive for the Preconditioner to remain valid (Positive Definite approx)
                # This is equivalent to saying the agent "trusts" the connection only when it's coherent
                stiffness = max(0.01, np.cos(diff)) 
                
                w_val = self.gamma * stiffness
                weights[i][j] = w_val
                diags[i] += w_val
                
        return force, diags, weights

    def solve_distributed(self, force, diags, weights, k_steps):
        if k_steps == 0:
            return force 
            
        inv_diag = 1.0 / diags
        u = force * inv_diag
        # Conservative relaxation for nonlinear stability
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

def run_nonlinear_criticality():
    N = 100
    COUPLING = 20.0 # Increased from 15.0
    STEPS = 1000     
    DT = 0.01
    
    # Reduced noise to allow synchronization to emerge
    T = 1.0 
    # Extended scan to find the higher threshold
    k_values = [0, 1, 2, 4, 8, 12, 16, 20, 24, 30] 
    
    results_order = []
    results_energy = []
    
    print(f"Scanning Nonlinear Criticality (N={N}, Gamma={COUPLING}, T={T})...")
    
    sys = NonlinearCriticalitySystem(n_nodes=N, coupling=COUPLING)
    
    for k in tqdm(k_values):
        # Random start (High Energy State)
        x = sys.mu + np.random.uniform(-np.pi, np.pi, N) 
        
        avg_order = 0
        avg_energy = 0
        samples = 0
        
        # Transient
        for _ in range(300):
            force, diags, weights = sys.get_force_and_weights(x)
            noise = np.random.normal(0, 1, N)
            
            if k == 0:
                dx = force * DT + noise * np.sqrt(2*T*DT)
            else:
                eff_force = sys.solve_distributed(force, diags, weights, k)
                eff_noise = sys.solve_distributed(noise, diags, weights, k//2)
                dx = eff_force * DT + eff_noise * np.sqrt(2*T*DT) * 2.0
            
            x += dx
            
        # Measurement
        for _ in range(STEPS - 300):
            force, diags, weights = sys.get_force_and_weights(x)
            noise = np.random.normal(0, 1, N)
            
            if k == 0:
                dx = force * DT + noise * np.sqrt(2*T*DT)
            else:
                eff_force = sys.solve_distributed(force, diags, weights, k)
                eff_noise = sys.solve_distributed(noise, diags, weights, k//2)
                dx = eff_force * DT + eff_noise * np.sqrt(2*T*DT) * 2.0
            
            x += dx
            
            avg_order += sys.get_order_parameter(x)
            avg_energy += sys.get_energy(x)
            samples += 1
            
        results_order.append(avg_order / samples)
        results_energy.append(avg_energy / samples)

    # Print Data
    print("\n--- NONLINEAR DATA REPORT ---")
    print(f"Temperature T={T}")
    print("K\tOrder(R)\tEnergy")
    for idx, k in enumerate(k_values):
        r_val = results_order[idx]
        e_val = results_energy[idx]
        print(f"{k}\t{r_val:.4f}\t{e_val:.4f}")

    # Plotting
    os.makedirs('figures', exist_ok=True)
    plt.style.use('bmh')
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Plot 1: Order
    ax1.plot(k_values, results_order, 'o-', color='#8e44ad', linewidth=2, markersize=8, label='Nonlinear (Kuramoto)')
    ax1.set_title('Nonlinear Phase Transition', fontsize=12)
    ax1.set_xlabel('Negotiation Depth K')
    ax1.set_ylabel('Order Parameter R')
    ax1.set_ylim(0, 1.05)
    ax1.axvspan(0.5, 3.5, color='gray', alpha=0.1, label='Critical Region?')
    ax1.legend()
    
    # Plot 2: Energy
    ax2.plot(k_values, results_energy, 's--', color='#8e44ad', linewidth=2, label='Nonlinear Energy')
    ax2.set_title('Thermodynamic Efficiency', fontsize=12)
    ax2.set_xlabel('Negotiation Depth K')
    ax2.set_ylabel('Energy (Log Scale)')
    ax2.set_yscale('log')
    ax2.legend()
    
    plt.tight_layout()
    plt.savefig('figures/nonlinear_criticality.png', dpi=300)
    print("Saved to figures/nonlinear_criticality.png")

if __name__ == "__main__":
    run_nonlinear_criticality()