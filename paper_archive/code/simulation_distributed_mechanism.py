import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from scipy.sparse.linalg import cg, LinearOperator
import os
from tqdm import tqdm

"""
EXP 20: Emergence of Geometry from Local Interactions
---------------------------------------------------
Addressing the "God View" criticism.
We demonstrate that the global Riemannian metric (Inverse Hessian) 
can naturally emerge from purely local, iterative interactions.

Core Mechanism:
Instead of computing G^{-1} explicitly (which requires global knowledge),
agents perform a fast local equilibration process (Communication/Diffusion)
before committing to a physical move.

This maps "Computational Time" (Communication steps K) to "Geometric Optimization".

Physics:
Solving (I + gamma*L) * u = Force  for u.
This is equivalent to finding the equilibrium of a fast diffusion field.
"""

class DistributedSystem:
    def __init__(self, n_nodes=100, k_neighbors=4, coupling=10.0, seed=42):
        self.N = n_nodes
        self.gamma = coupling
        np.random.seed(seed)
        
        # 1. Topology: Ring Lattice with shortcuts (Small World)
        # Represents a realistic supply chain or neural network
        self.G = nx.watts_strogatz_graph(n_nodes, k=k_neighbors, p=0.05, seed=seed)
        self.L = nx.laplacian_matrix(self.G).toarray()
        self.A = nx.adjacency_matrix(self.G).toarray()
        self.D = np.diag(np.sum(self.A, axis=1))
        
        # Hessian H = 2(I + gamma*L)
        # We need to solve H * u = v  <=>  (I + gamma*L) * u = v/2
        self.H = 2.0 * (np.eye(self.N) + self.gamma * self.L)
        
        # Target State (Ground Truth) - High frequency variation to challenge the system
        # self.mu = np.random.uniform(-5, 5, n_nodes)
        # A mix of smooth gradient and random spikes
        xs = np.linspace(0, 4*np.pi, n_nodes)
        self.mu = 5 * np.sin(xs) + np.random.normal(0, 1, n_nodes)

    def get_energy(self, x):
        return np.sum((x - self.mu)**2) + self.gamma * (x.T @ self.L @ x)

    def get_force(self, x):
        # F = - Gradient
        return - (2.0 * (x - self.mu) + 2.0 * self.gamma * (self.L @ x))

    def solve_distributed_tensor_update(self, force, k_communication_steps):
        """
        The KEY Innovation: Distributed Approximation of G^{-1}
        
        We want to find u such that: H * u = force
        2 * (I + gamma * L) * u = force
        
        This is a linear system Ax = b.
        We can solve it using Jacobi Iteration or Richardson Iteration,
        which are purely local operations (node only talks to neighbors).
        
        Richardson Iteration:
        u_{t+1} = u_t - alpha * (H * u_t - force)
        
        This represents 'k' rounds of message passing between neighbors.
        """
        # Initial guess (naive response = local gradient)
        # Scaling factor 1/2 comes from the diagonal of H being approx 2(1+gamma*deg)
        # A simple approximation for the diagonal preconditioner
        diag_H = 2.0 * (1.0 + self.gamma * np.diag(self.D))
        u = force / diag_H # Jacobi preconditioner start
        
        # Iterative refinement (Communication rounds)
        # This simulates the "Thinking" or "Negotiation" phase
        # Ensure alpha is small enough for convergence: alpha < 2/lambda_max
        # lambda_max approx 2 * (1 + 2*gamma*k)
        max_eigenval = 2.0 * (1.0 + self.gamma * 2.0 * 4.0) # Approx upper bound
        alpha = 0.9 / max_eigenval
        
        for _ in range(k_communication_steps):
            # Calculate H * u locally
            # L * u requires only neighbor info
            Lu = self.L @ u
            Hu = 2.0 * (u + self.gamma * Lu)
            
            # Residual error
            r = force - Hu
            
            # Update guess
            u = u + alpha * r
            
        return u

    def ideal_tensor_update(self, force):
        # Cheating: Global Inversion
        return np.linalg.solve(self.H, force)

def run_experiment():
    N = 100
    COUPLING = 15.0 # Slightly reduced for stability
    steps = 1500
    dt = 0.002 # Reduced dt for numerical stability
    
    sys = DistributedSystem(n_nodes=N, coupling=COUPLING)
    
    # Scenarios: Different levels of "Cognitive Depth" (K steps)
    k_steps_list = [0, 1, 5, 20] 
    results = {}
    
    print(f"Simulating Distributed Emergence (N={N}, Coupling={COUPLING})...")
    
    for k in k_steps_list:
        print(f"Running for Communication Depth K={k}...")
        x = np.zeros(N)
        errors = []
        
        for _ in range(steps):
            force = sys.get_force(x)
            
            # Add thermal noise (Langevin)
            noise = np.random.normal(0, 1, N)
            
            if k == 0:
                # Pure Scalar (Naive)
                # Just follow the force directly (overdamped)
                dx = force * dt + noise * np.sqrt(dt) 
            else:
                # Emergent Tensor (Distributed Negotiation)
                # 1. Filter the force through the network (Noise reduction + Coordination)
                effective_force = sys.solve_distributed_tensor_update(force, k_communication_steps=k)
                
                # 2. Filter the noise too (Geometric Brownian Motion)
                # To satisfy Fluctuation-Dissipation Theorem properly, noise should also be colored.
                # Here we approximate it by smoothing the noise too.
                effective_noise = sys.solve_distributed_tensor_update(noise, k_communication_steps=k//2) 
                # Note: Sqrt(G^-1) is roughly G^-1/2, so half steps for noise
                
                dx = effective_force * dt + effective_noise * np.sqrt(dt) * 5.0 # Scale noise for visibility
            
            x += dx
            
            # Record Energy/Error
            errors.append(sys.get_energy(x))
            
        results[k] = errors

    # Run Ideal Benchmark
    print("Running Ideal Tensor Benchmark...")
    x = np.zeros(N)
    ideal_errors = []
    for _ in range(steps):
        force = sys.get_force(x)
        noise = np.random.normal(0, 1, N)
        
        # Ideal Newton Step
        dx = sys.ideal_tensor_update(force) * dt 
        # Ideal Noise (Correct FDT)
        # For visualization simply, we skip complex noise generation for ideal line here 
        # or use a simplified version
        dx += noise * np.sqrt(dt) * 0.1 
        
        x += dx
        ideal_errors.append(sys.get_energy(x))

    # Plotting
    plt.figure(figsize=(10, 6))
    plt.style.use('bmh')
    
    colors = {0: '#e74c3c', 1: '#e67e22', 5: '#f1c40f', 20: '#2ecc71'}
    labels = {
        0: 'Scalar (K=0, Reflexive)',
        1: 'Local (K=1, Nearest Neighbor)',
        5: 'Meso (K=5, Short-range)',
        20: 'Global-like (K=20, Long-range)'
    }
    
    for k in k_steps_list:
        plt.plot(results[k], label=labels[k], color=colors[k], alpha=0.9, linewidth=1.5)
        
    plt.plot(ideal_errors, 'k--', label='Ideal Tensor (God View)', alpha=0.5)
    
    plt.title('Emergence of Geometric Coordination from Local Communication', fontsize=12)
    plt.xlabel('Simulation Steps (Time)')
    plt.ylabel('System Free Energy (Log Scale)')
    plt.yscale('log')
    plt.legend()
    plt.grid(True, which='both', alpha=0.3)
    
    os.makedirs('figures', exist_ok=True)
    save_path = 'figures/distributed_emergence.png'
    plt.savefig(save_path, dpi=300)
    print(f"Saved figure to {save_path}")

if __name__ == "__main__":
    run_experiment()