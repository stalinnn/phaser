import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from tqdm import tqdm
import os

"""
EXP 70: Thermodynamic Attention Dynamics
-------------------------------------------------------
A principled resolution to the "Nonlinear Limit".

Hypothesis:
The failure of fixed coupling (or simple clipping) is due to the inability
to dynamicallly re-route information flow around geometric singularities.
We introduce a "Thermodynamic Attention" mechanism where agents assign 
weights (Attention) to neighbors based on their local consistency.

Mechanism:
Weights w_ij are computed via a Softmax over the "Energy Gap" (Hessian reliability).
This creates a smooth, differentiable gating mechanism that naturally handles
saddle points by down-weighting them (low attention) relative to the Anchor.
"""

class AttentionSystem:
    def __init__(self, n_nodes=100, k_neighbors=4, coupling=20.0, seed=42):
        self.N = n_nodes
        self.gamma = coupling
        np.random.seed(seed)
        
        self.G = nx.watts_strogatz_graph(n_nodes, k=k_neighbors, p=0.1, seed=seed)
        self.adj_list = [list(self.G.neighbors(i)) for i in range(n_nodes)]
        
        xs = np.linspace(0, 2*np.pi, n_nodes)
        self.mu = xs 
        
    def get_order_parameter(self, x):
        deviations = x - self.mu
        z = np.mean(np.exp(1j * deviations))
        return np.abs(z)

    def get_force_and_attention(self, x):
        force = np.zeros(self.N)
        force -= (x - self.mu)
        
        diags = np.zeros(self.N) 
        weights = [{} for _ in range(self.N)]
        
        # Temperature for the Attention Softmax
        # If too low -> Hard switching (Greedy)
        # If too high -> Uniform average
        T_attention = 0.1 
        
        for i in range(self.N):
            # 1. Calculate "Logits" (Quality Scores) for each source
            # Source 0: Anchor (Self)
            # Source j: Neighbors
            
            # Score metric: Cosine Similarity (Alignment)
            # High alignment = High Trust
            
            # Anchor score (Always reliable, but maybe not informative enough)
            logits = {}
            logits['self'] = 1.0 # Baseline trust
            
            neighbor_indices = self.adj_list[i]
            for j in neighbor_indices:
                diff = x[i] - x[j]
                # Logit is proportional to stability (cos(theta))
                logits[j] = np.cos(diff)
            
            # 2. Compute Softmax Attention Weights
            # max_logit for numerical stability
            max_l = max(logits.values())
            sum_exp = 0
            exps = {}
            
            for key, val in logits.items():
                e = np.exp((val - max_l) / T_attention)
                exps[key] = e
                sum_exp += e
                
            # 3. Assign Weights
            # The total "Budget" of stiffness is (1 + Gamma * k)
            # We redistribute this budget according to Attention
            
            total_stiffness = 1.0 + self.gamma * len(neighbor_indices)
            
            # Anchor contribution
            alpha_self = exps['self'] / sum_exp
            diags[i] += total_stiffness * alpha_self
            
            # Neighbor contributions
            for j in neighbor_indices:
                alpha_j = exps[j] / sum_exp
                
                # The effective weight is Total_Stiffness * Attention_j
                w_val = total_stiffness * alpha_j
                
                weights[i][j] = w_val
                diags[i] += w_val
                
                # Physics Force is unchanged (Gradient of original Potential)
                diff = x[i] - x[j]
                force[i] -= self.gamma * np.sin(diff)
                
        return force, diags, weights

    def solve_distributed(self, force, diags, weights, k_steps):
        if k_steps == 0:
            return force 
        inv_diag = 1.0 / diags
        u = force * inv_diag
        alpha = 0.8 # Higher alpha possible due to stability
        
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

def run_attention_experiment():
    N = 100
    COUPLING = 20.0 
    STEPS = 1000     
    DT = 0.01
    T = 1.0 
    
    k_values = [0, 4, 16, 32, 64]
    results = []
    
    sys = AttentionSystem(n_nodes=N, coupling=COUPLING)
    
    print(f"Running Thermodynamic Attention Dynamics (N={N})...")
    
    for k in tqdm(k_values):
        np.random.seed(42) 
        x = sys.mu + np.random.uniform(-np.pi, np.pi, N) 
        
        avg_order = 0
        
        for t in range(STEPS):
            force, diags, weights = sys.get_force_and_attention(x)
            noise = np.random.normal(0, 1, N)
            
            if k == 0:
                dx = force * DT + noise * np.sqrt(2*T*DT)
            else:
                eff_force = sys.solve_distributed(force, diags, weights, k)
                eff_noise = sys.solve_distributed(noise, diags, weights, k//2)
                dx = eff_force * DT + eff_noise * np.sqrt(2*T*DT) * 2.0
            
            x += dx
            
            if t > 500:
                avg_order += sys.get_order_parameter(x)
        
        results.append(avg_order / 500)

    print("\n--- ATTENTION RESULTS ---")
    print("K\tOrder(R)")
    for i, k in enumerate(k_values):
        print(f"{k}\t{results[i]:.4f}")

    os.makedirs('figures', exist_ok=True)
    plt.style.use('bmh')
    plt.figure(figsize=(8, 6))
    plt.plot(k_values, results, 'o-', color='#e74c3c', linewidth=3, label='Thermodynamic Attention')
    plt.axhline(y=0.8, color='gray', linestyle='--')
    plt.title('Thermodynamic Attention Mechanism', fontsize=12)
    plt.xlabel('Negotiation Depth K')
    plt.ylabel('Order Parameter R')
    plt.savefig('figures/attention_fix.png', dpi=300)

if __name__ == "__main__":
    run_attention_experiment()
