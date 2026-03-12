import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from scipy.sparse.linalg import cg
import os
from tqdm import tqdm

"""
EXP 30: Nonlinear Dynamics on Curved Manifolds (Kuramoto/XY Model)
----------------------------------------------------------------
Here we introduce nonlinearity. The coupling is no longer a simple spring,
but a sinusoidal potential (Kuramoto model).
V(x) = 0.5 * sum(x_i - mu_i)^2 + gamma * sum (1 - cos(x_i - x_j))

The Hessian (Metric Tensor) is now state-dependent:
H_ij(x) depends on cos(x_i - x_j).

This tests if the "Distributed Geometric Mechanics" holds in non-Euclidean regimes.
"""

class NonlinearSystem:
    def __init__(self, n_nodes=100, k_neighbors=4, coupling=10.0, seed=42):
        self.N = n_nodes
        self.gamma = coupling
        np.random.seed(seed)
        
        # Topology: Small World Ring
        self.G = nx.watts_strogatz_graph(n_nodes, k=k_neighbors, p=0.05, seed=seed)
        self.adj_list = [list(self.G.neighbors(i)) for i in range(n_nodes)]
        
        # Target State (Ground Truth)
        xs = np.linspace(0, 4*np.pi, n_nodes)
        self.mu = 3 * np.sin(xs) + np.random.normal(0, 0.5, n_nodes)

    def get_energy(self, x):
        # Anchor energy
        e_anchor = 0.5 * np.sum((x - self.mu)**2)
        
        # Interaction energy (XY Model / Kuramoto)
        e_inter = 0
        for i in range(self.N):
            for j in self.adj_list[i]:
                if i < j: # Count each edge once
                    e_inter += self.gamma * (1 - np.cos(x[i] - x[j]))
        return e_anchor + e_inter

    def get_force_and_curvature(self, x):
        """
        Returns:
        1. Force vector (Negative Gradient)
        2. Local curvature info (Diagonal and Weighted Adjacency for Laplacian)
        """
        force = np.zeros(self.N)
        # We need to construct the operator H(x) implicitly
        # H_ii = 1 + gamma * sum_j cos(diff)
        # H_ij = -gamma * cos(diff)
        
        # Anchor term
        force -= (x - self.mu)
        
        # Interaction term
        # For implicit solver, we store the weighted adjacency weights
        # weights[i][neighbor_index] = cos(x_i - x_j)
        weights = [] 
        diags = np.ones(self.N) # Start with identity from anchor
        
        for i in range(self.N):
            w_i = {}
            for j in self.adj_list[i]:
                diff = x[i] - x[j]
                
                # Gradient contribution: - sin(diff)
                # Force += - (-gamma * sin(diff)) = gamma * sin(diff) (pulls towards j)
                force[i] -= self.gamma * np.sin(diff)
                
                # Hessian contribution: cos(diff)
                cos_d = np.cos(diff)
                
                # Robustness trick: If cos_d is negative (unstable/repulsive), 
                # we clip it to small positive or zero for the conditioner to avoid divergence.
                # This effectively means "trust the anchor more when neighbors disagree wildly"
                effective_stiffness = max(0.01, cos_d) 
                
                w_i[j] = self.gamma * effective_stiffness
                diags[i] += self.gamma * effective_stiffness
                
            weights.append(w_i)
            
        return force, diags, weights

    def solve_distributed_update(self, force, diags, weights, k_steps):
        """
        Solves H(x) * u = force using distributed Jacobi/Richardson iteration.
        Also interpreted as Real-space Renormalization Group (RG) Flow.
        
        Physics:
        Each iteration step effectively coarse-grains the system, filtering out
        high-frequency noise and expanding the effective correlation length.
        """
        # Preconditioner (Inverse Diagonal) - Local Approximation
        inv_diag = 1.0 / diags
        
        # Initial guess: u_0 = D^{-1} * force (Local Response)
        u = force * inv_diag
        
        # Relaxation parameter (Learning Rate for the Inner Loop)
        alpha = 0.5 
        
        # Iterative RG Flow (K steps)
        for _ in range(k_steps):
            # Compute H * u matrix-vector product locally
            # (H*u)_i = diags[i]*u_i - sum_{j} w_{ij} * u_j
            Hu = np.zeros(self.N)
            for i in range(self.N):
                sum_neighbors = 0
                for j, w_val in weights[i].items():
                    sum_neighbors += w_val * u[j]
                
                Hu[i] = diags[i] * u[i] - sum_neighbors
            
            # Residual (Force Imbalance)
            r = force - Hu
            
            # Update State (Flow towards Fixed Point)
            u = u + alpha * r * inv_diag 
            
        return u

def run_nonlinear_experiment():
    N = 100
    COUPLING = 8.0 
    steps = 2000
    dt = 0.005 
    
    sys = NonlinearSystem(n_nodes=N, coupling=COUPLING)
    
    k_steps_list = [0, 5, 20]
    results = {}
    
    print(f"Simulating Nonlinear (Kuramoto) Dynamics (N={N})...")
    
    for k in k_steps_list:
        print(f"Running K={k}...")
        x = np.random.uniform(-np.pi, np.pi, N) # Random start (high energy)
        errors = []
        
        for _ in tqdm(range(steps)):
            # 1. Get Force and Instantaneous Curvature (State-dependent)
            force, diags, weights = sys.get_force_and_curvature(x)
            
            # 2. Add Noise
            noise = np.random.normal(0, 1, N)
            
            if k == 0:
                dx = force * dt + noise * np.sqrt(dt)
            else:
                # Apply Geometric Correction
                effective_force = sys.solve_distributed_update(force, diags, weights, k)
                effective_noise = sys.solve_distributed_update(noise, diags, weights, k//2)
                dx = effective_force * dt + effective_noise * np.sqrt(dt) * 3.0
            
            x += dx
            errors.append(sys.get_energy(x))
            
        results[k] = errors

    # Plotting
    plt.figure(figsize=(10, 6))
    plt.style.use('bmh')
    
    for k in k_steps_list:
        plt.plot(results[k], label=f'K={k} ({"Scalar" if k==0 else "Tensor"})', alpha=0.8)
        
    plt.title('Nonlinear Coordination: Escaping the Complexity Trap', fontsize=12)
    plt.xlabel('Time Steps')
    plt.ylabel('Potential Energy V(x)')
    plt.yscale('log')
    plt.legend()
    
    os.makedirs('figures', exist_ok=True)
    plt.savefig('figures/nonlinear_dynamics.png', dpi=300)
    print("Saved to figures/nonlinear_dynamics.png")

if __name__ == "__main__":
    run_nonlinear_experiment()