import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from simulation import EconomySimulationBase, SmartMarketSimulation, TensorSynergySimulation
import scipy.sparse as sp
import time

def run_smart_phase_diagram_scan():
    """
    Generate a Phase Diagram: Complexity (N) vs. Volatility (Noise).
    This compares Tensor Coordination vs. SMART Market (with Arbitrage & Momentum).
    """
    print("=== Initiating Phase Diagram Scan (Tensor vs Smart Market) ===")
    
    # Parameters for the Phase Diagram
    # We suspect Tensor wins at High Complexity (N) and High Noise (Volatility)
    # But Smart Market is very good at Low Complexity.
    n_agents_list = np.linspace(50, 500, 6, dtype=int)  # X-axis: System Complexity
    noise_levels = np.linspace(0.1, 8.0, 6)            # Y-axis: Environmental Volatility
    
    # Grid to store the relative advantage
    advantage_grid = np.zeros((len(noise_levels), len(n_agents_list)))
    
    steps = 80
    trials_per_point = 5 # Increase trials for robustness
    
    total_iterations = len(n_agents_list) * len(noise_levels)
    current_iter = 0
    start_time = time.time()

    for i, noise in enumerate(noise_levels):
        for j, n_agents in enumerate(n_agents_list):
            current_iter += 1
            print(f"[{current_iter}/{total_iterations}] Simulating N={n_agents}, Noise={noise:.2f}...")
            
            trial_advantages = []
            
            for _ in range(trials_per_point):
                # Shared Base State
                sim_base = EconomySimulationBase(n_agents=n_agents, graph_type='scale_free')
                
                # 1. Smart Market Simulation (Steel-manned Opponent)
                market_sim = SmartMarketSimulation(n_agents=n_agents)
                market_sim.G_dense = sim_base.G_dense.copy()
                market_sim.G_sparse = sp.csr_matrix(market_sim.G_dense) # Needed for diffusion
                market_sim.x = sim_base.x.copy()
                market_sim.x_target = sim_base.x_target.copy()
                # Enable arbitrage and momentum
                market_sim.run(steps=steps, dt=0.02, shock_at=steps//2, shock_mag=noise, 
                               arbitrage_coeff=0.8, momentum_coeff=0.2, volatility=noise*0.05)
                
                # Calculate final error (steady state)
                market_error = np.mean(market_sim.history['error'][-15:])
                
                # 2. Tensor Simulation (Our Model)
                tensor_sim = TensorSynergySimulation(n_agents=n_agents)
                tensor_sim.G_dense = sim_base.G_dense.copy()
                tensor_sim.G_sparse = sp.csr_matrix(tensor_sim.G_dense)
                tensor_sim.D_inv = sim_base.D_inv
                tensor_sim.R_sparse = sim_base.R_sparse
                tensor_sim.x = sim_base.x.copy()
                tensor_sim.x_target = sim_base.x_target.copy()
                tensor_sim.run(steps=steps, dt=0.05, shock_at=steps//2, shock_mag=noise, 
                               bp_steps=5, volatility=noise*0.05)
                
                tensor_error = np.mean(tensor_sim.history['error'][-15:])
                
                # Relative Advantage Metric
                # If market_error is small, this can explode, so we clip denominator
                denom = max(market_error, 1e-6)
                advantage = (market_error - tensor_error) / denom
                trial_advantages.append(advantage)
            
            advantage_grid[i, j] = np.mean(trial_advantages)

    elapsed = time.time() - start_time
    print(f"Scan completed in {elapsed:.2f} seconds.")

    # Plotting
    plt.figure(figsize=(10, 8))
    sns.set_style("whitegrid")
    
    # Create Heatmap
    ax = sns.heatmap(advantage_grid, xticklabels=n_agents_list, yticklabels=np.round(noise_levels, 1),
                     cmap="RdBu_r", center=0, annot=True, fmt=".2f", 
                     cbar_kws={'label': 'Tensor Advantage over Smart Market (>0 means Tensor wins)'})
    
    ax.invert_yaxis() # Put low noise at bottom
    plt.title('Phase Diagram: Smart Market vs. Tensor Network\n(Search for the Phase Transition Boundary)', fontsize=14)
    plt.xlabel('System Complexity (N)', fontsize=12)
    plt.ylabel('Volatility / Noise', fontsize=12)
    
    output_path = 'figures/simulation_phase_diagram_smart.png'
    # Ensure directory exists
    import os
    if not os.path.exists('figures'):
        os.makedirs('figures')
        
    plt.savefig(output_path, dpi=300)
    print(f"Phase Diagram saved to {output_path}")

if __name__ == "__main__":
    run_smart_phase_diagram_scan()

