import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import os
import time
from tqdm import tqdm

"""
EXP 12: Finite Size Scaling (FSS) Analysis
------------------------------------------
REVISED (Honest Implementation):
We verify if "Distributed Tensor Dynamics" can overcome the complexity trap 
WITHOUT assuming a global God-view (matrix inversion).

We compare:
1. Scalar Dynamics (Gradient Descent): Suffering from Critical Slowing Down.
2. Distributed Tensor (Riemannian Langevin): Using purely local negotiation (K steps).

Key Hypothesis:
If Communication Depth (K) scales linearly with System Size (N), 
the coordination error should remain constant, avoiding the power-law divergence.
"""

class NetworkHamiltonian:
    def __init__(self, n_nodes, connectivity=2, coupling_strength=1.0, seed=42):
        self.N = n_nodes
        self.gamma = coupling_strength
        np.random.seed(seed)
        
        # 1. Topology: 1D Ring Lattice
        # This is the "Hard Mode" for coordination due to large diameter (D ~ N/2)
        self.G_graph = nx.watts_strogatz_graph(n_nodes, k=2, p=0.0, seed=seed)
        self.L = nx.laplacian_matrix(self.G_graph).toarray()
        
        # 2. Target State with Long-range Gradient
        # Forces information to propagate across the entire network
        self.mu = np.linspace(-5, 5, n_nodes) 
        self.mu += np.random.normal(0, 0.5, n_nodes)
        
        # 3. Spectral bounds for Solver Stability
        # Max eigenvalue of 1D Laplacian is 4.0
        # H = 2(I + gamma*L) => Max Eig = 2(1 + 4*gamma)
        self.max_eig_H = 2.0 * (1.0 + self.gamma * 4.0)

    def gradient(self, x):
        return 2.0 * (x - self.mu) + 2.0 * self.gamma * (self.L @ x)

    def solve_distributed(self, v, k_steps):
        """
        Solves H * u = v using purely LOCAL Richardson Iteration.
        REPLACES the global 'inv(H)' to ensure physical realism.
        
        RG Perspective:
        Real-space renormalization flow.
        k_steps determines the effective correlation length xi ~ k * a.
        """
        # Preconditioner (Jacobi): Inverse Diagonal of H
        # H_ii = 2 * (1 + gamma * degree_i)
        diags = 2.0 * (1.0 + self.gamma * self.L.diagonal())
        u = v / diags 
        
        # Relaxation parameter (must be < 2/lambda_max)
        alpha = 1.9 / self.max_eig_H
        
        for _ in range(k_steps):
            # Local Operation: H * u
            # (L @ u) is just summing differences with neighbors
            Hu = 2.0 * (u + self.gamma * (self.L @ u))
            r = v - Hu
            u = u + alpha * r
            
        return u

def run_simulation(N, steps=2000, dt=0.01, T=1.0, trials=5):
    """
    Runs comparison for system size N.
    """
    scalar_errors = []
    tensor_errors = []
    
    # HONESTY CONFIG:
    # For Tensor Dynamics to work on a ring, information must travel the diameter.
    # Diameter ~ N/2. So we set negotiation steps K ~ N.
    # This represents "Linear Computational Cost" for "Constant Control Error".
    K_steps = int(N * 0.5) + 5
    
    for _ in range(trials):
        sys = NetworkHamiltonian(n_nodes=N)
        x0 = np.zeros(N) 
        sq_dt = np.sqrt(dt)
        sqrt_2T = np.sqrt(2 * T)
        
        # --- 1. Scalar Run (Naive Gradient) ---
        x = x0.copy()
        for _ in range(steps):
            drift = -sys.gradient(x)
            noise = np.random.normal(0, 1, N)
            x += drift * dt + sqrt_2T * noise * sq_dt
        scalar_errors.append(np.mean((x - sys.mu)**2))
        
        # --- 2. Tensor Run (Distributed Negotiation) ---
        x = x0.copy()
        for _ in range(steps):
            grad = sys.gradient(x)
            
            # Replaced global inverse with local solver
            drift = -sys.solve_distributed(grad, k_steps=K_steps)
            
            # Colored Noise Approximation (Fluctuation-Dissipation)
            raw_noise = np.random.normal(0, 1, N)
            noise = sys.solve_distributed(raw_noise, k_steps=K_steps//2)
            
            x += drift * dt + sqrt_2T * noise * sq_dt
            
        tensor_errors.append(np.mean((x - sys.mu)**2))
        
    return np.mean(scalar_errors), np.std(scalar_errors), np.mean(tensor_errors), np.std(tensor_errors)

def main():
    # Reduced max size slightly for speed since we use iterative solver now
    sizes = [20, 40, 60, 80, 100, 150] 
    
    results = {
        'N': sizes,
        'scalar_mean': [], 'scalar_std': [],
        'tensor_mean': [], 'tensor_std': []
    }
    
    print("Starting HONEST Finite Size Scaling (FSS) Analysis...")
    print("Comparing Scalar Diffusion vs. Distributed Negotiation (K ~ N/2)")
    
    start_time = time.time()
    
    for N in tqdm(sizes, desc="Scaling System Size"):
        s_mean, s_std, t_mean, t_std = run_simulation(N, trials=8, T=1.0)
        
        results['scalar_mean'].append(s_mean)
        results['scalar_std'].append(s_std)
        results['tensor_mean'].append(t_mean)
        results['tensor_std'].append(t_std)
        
    print(f"\nSimulation finished in {time.time()-start_time:.1f}s")
    
    # --- Plotting ---
    os.makedirs('figures', exist_ok=True)
    plt.style.use('bmh')
    
    fig, ax = plt.subplots(figsize=(10, 7))
    
    # Scalar Curve
    ax.errorbar(sizes, results['scalar_mean'], yerr=results['scalar_std'], 
               fmt='o--', color='#e74c3c', label='Scalar (Local Gradient)', capsize=4, linewidth=2)
    
    # Tensor Curve
    ax.errorbar(sizes, results['tensor_mean'], yerr=results['tensor_std'], 
               fmt='s-', color='#2ecc71', label='Tensor (Distributed K~N)', capsize=4, linewidth=2)
    
    # Theoretical Scaling Lines
    n_arr = np.array(sizes)
    scale_factor = results['scalar_mean'][0] / (n_arr[0]**1.5) 
    ax.plot(n_arr, scale_factor * (n_arr**1.5), 'r:', alpha=0.5, label='Theory: Diffusive ($\sim N^{1.5}$)')

    ax.set_xscale('log')
    ax.set_yscale('log') 
    
    ax.set_xlabel('System Size $N$ (Ring Topology)', fontsize=12)
    ax.set_ylabel('Coordination Error (MSE)', fontsize=12)
    ax.set_title('Overcoming the Complexity Trap via Local Negotiation', fontsize=14)
    
    ax.legend(fontsize=11)
    ax.grid(True, which="both", ls="-", alpha=0.4)
    
    plt.tight_layout()
    plt.savefig('figures/finite_size_scaling_ring.png', dpi=300)
    print("Saved results to figures/finite_size_scaling_ring.png")

if __name__ == "__main__":
    main()