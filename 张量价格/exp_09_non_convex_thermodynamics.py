import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from simulation import EconomySimulationBase
import scipy.sparse as sp

# --- 定义非凸势能面 (Non-Convex Landscape) ---
def double_well_potential(x, target, depth=3.0, barrier_width=1.0, bias=2.0):
    """
    非对称双阱势能：V(u) = depth * (u^2 - 1)^2 - bias * u
    bias > 0 会让右边的坑(x=+1)更深，成为 Global Minimum。
    左边的坑(x=-1)变成 Metastable (亚稳态/局部最优)。
    """
    u = (x - target) / barrier_width
    # Clip u to avoid overflow in visualization if x explodes
    u = np.clip(u, -5, 5) 
    return depth * (u**2 - 1)**2 - bias * u

def double_well_force(x, target, depth=3.0, barrier_width=1.0, bias=2.0):
    """
    Force = -grad V
    """
    u = (x - target) / barrier_width
    # Clip u for numerical stability of forces
    u = np.clip(u, -3, 3)
    # F = - (dV/du) * (du/dx)
    # dV/du = 4*depth*(u^3 - u) - bias
    force_u = -(4 * depth * (u**3 - u) - bias)
    return force_u / barrier_width

class NonConvexSimulation:
    def __init__(self, n_agents=100, T=0.8):
        np.random.seed(42)
        self.N = n_agents
        self.T = T 
        
        # Start near the "Bad" Local Optimum (-1)
        self.x_m = np.random.normal(-1.0, 0.1, self.N)
        self.x_t = np.random.normal(-1.0, 0.1, self.N)
        self.target = np.zeros(self.N) 
        
        # Topology for Tensor
        import networkx as nx
        self.G_graph = nx.watts_strogatz_graph(n_agents, k=4, p=0.1)
        self.adj_matrix = nx.to_numpy_array(self.G_graph)
        # Normalize adjacency to avoid force explosion for high-degree nodes
        degree = np.sum(self.adj_matrix, axis=1)
        with np.errstate(divide='ignore', invalid='ignore'):
            self.norm_adj = self.adj_matrix / degree[:, None]
            self.norm_adj[np.isnan(self.norm_adj)] = 0
            
        self.history = {'market_energy': [], 'tensor_energy': [], 'market_x': [], 'tensor_x': []}

    def run_comparison(self, steps=1000, dt=0.05):
        print(f"Running Asymmetric Non-Convex Competition (Bias=2.0, Market T={self.T}, Tensor T={self.T*0.1})...")
        
        for t in range(steps):
            # --- 1. Market: High Temperature Langevin ---
            # Can jump over barriers
            f_m = double_well_force(self.x_m, self.target)
            noise_m = np.random.normal(0, 1, self.N)
            # dx = F*dt + sigma*dW
            dx_m = f_m * dt + np.sqrt(2 * self.T * dt) * noise_m
            self.x_m += dx_m
            
            # --- 2. Tensor Network: "Groupthink" / Over-coordination ---
            # The tensor network smooths out the gradient noise, making it efficient locally
            # but hard to escape globally because "neighbors pull you back".
            
            f_t = double_well_force(self.x_t, self.target)
            
            # Coupling: "If my neighbors feel a force, I assume I should too."
            # This is the essence of Belief Propagation: sharing gradient info.
            neighbor_avg_force = self.norm_adj.dot(f_t)
            
            # Tensor logic: rely heavily on consensus (neighbor force)
            # Alpha = 0.8 means 80% of decision is based on social proof
            # This strong coupling prevents individual "mutations" (tunneling)
            effective_force = 0.2 * f_t + 0.8 * neighbor_avg_force
            
            # Low internal noise (Rigid system)
            T_tensor = self.T * 0.05 
            noise_t = np.random.normal(0, 1, self.N)
            
            dx_t = effective_force * dt + np.sqrt(2 * T_tensor * dt) * noise_t
            self.x_t += dx_t
            
            # --- Recording ---
            E_m = np.mean(double_well_potential(self.x_m, self.target))
            E_t = np.mean(double_well_potential(self.x_t, self.target))
            
            self.history['market_energy'].append(E_m)
            self.history['tensor_energy'].append(E_t)
            self.history['market_x'].append(np.mean(self.x_m))
            self.history['tensor_x'].append(np.mean(self.x_t))
            
    def plot_results(self):
        plt.style.use('seaborn-v0_8-whitegrid')
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 12))
        
        steps = range(len(self.history['market_x']))
        
        # Plot 1: Trajectory (Tunneling Event)
        ax1.plot(steps, self.history['market_x'], 'r-', alpha=0.8, linewidth=1.5, label='Market (High Entropy)')
        ax1.plot(steps, self.history['tensor_x'], 'b-', alpha=0.8, linewidth=2.0, label='Tensor (Low Entropy)')
        
        # Add Landscape annotation
        ax1.axhline(y=1.0, color='g', linestyle='--', linewidth=2, label='Global Optimum (+1)')
        ax1.axhline(y=-1.0, color='k', linestyle=':', linewidth=2, label='Local Optimum (-1)')
        ax1.fill_between(steps, -0.2, 0.2, color='gray', alpha=0.2, label='Energy Barrier')
        
        ax1.set_ylabel('Average State <x>')
        ax1.set_title('The Trap of Efficiency: Market Finds Better Global Min due to Noise', fontsize=14)
        ax1.legend(loc='upper left')
        ax1.set_ylim(-1.5, 1.5)
        
        # Plot 2: Free Energy Landscape
        # Calculate moving average to smooth out thermal noise for clearer visualization
        def moving_average(a, n=10) :
            ret = np.cumsum(a, dtype=float)
            ret[n:] = ret[n:] - ret[:-n]
            return ret[n - 1:] / n

        m_energy_smooth = moving_average(self.history['market_energy'], n=20)
        t_energy_smooth = moving_average(self.history['tensor_energy'], n=20)
        valid_steps = steps[len(steps)-len(m_energy_smooth):]
        
        ax2.plot(valid_steps, m_energy_smooth, 'r-', label='Market Potential Energy')
        ax2.plot(valid_steps, t_energy_smooth, 'b-', label='Tensor Potential Energy')
        
        ax2.set_ylabel('Mean Potential Energy V(x)')
        ax2.set_xlabel('Time Steps')
        ax2.set_yscale('log')
        ax2.set_title('Thermodynamic Cost: Market Pays for Exploration', fontsize=12)
        ax2.legend()
        
        plt.tight_layout()
        plt.savefig('figures/non_convex_result.png', dpi=300)
        print("Saved to figures/non_convex_result.png")

if __name__ == "__main__":
    # T needs to be high enough to jump barrier depth=3.0
    # Kramer rate ~ exp(-Depth/T). If T=0.8, exp(-3/0.8) ~ exp(-3.75) ~ 0.02
    # In 1000 steps, probability of jump is high.
    sim = NonConvexSimulation(n_agents=200, T=0.9)
    sim.run_comparison(steps=1500, dt=0.05)
    sim.plot_results()
