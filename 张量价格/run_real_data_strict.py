import numpy as np
import matplotlib.pyplot as plt
import scipy.sparse as sp
from simulation import MarketScalarSimulation, TensorSynergySimulation, EconomySimulationBase
import os
import pickle

GRAPH_FILE = "real_data/bitcoin_alpha_graph.gpickle"

def run_real_world_strict_analysis():
    print(f"=== Running STRICT Real-World Analysis on Bitcoin Alpha ===")
    
    if not os.path.exists(GRAPH_FILE):
        print(f"Error: {GRAPH_FILE} not found. Please run download_and_prep_data.py first.")
        return

    # 1. Load Real Topology
    print("Loading Bitcoin Alpha topology...")
    # Initialize base to setup G_dense based on TRUST weights
    sim_base = EconomySimulationBase(graph_type=GRAPH_FILE)
    N = sim_base.N
    print(f"Network loaded: N={N} agents.")
    
    STEPS = 80
    
    # 2. Run Market Simulation (Baseline)
    market_sim = MarketScalarSimulation(n_agents=N, graph_type=GRAPH_FILE)
    # Inherit state variables for fair comparison
    market_sim.G_dense = sim_base.G_dense.copy()
    market_sim.x = sim_base.x.copy()
    market_sim.x_target = sim_base.x_target.copy()
    
    market_sim.run(steps=STEPS, dt=0.02)
    
    # 3. Run Tensor Simulation (Belief Propagation Mode)
    tensor_sim = TensorSynergySimulation(n_agents=N, graph_type=GRAPH_FILE)
    # Inherit state
    tensor_sim.G_dense = sim_base.G_dense.copy()
    tensor_sim.G_sparse = sp.csr_matrix(tensor_sim.G_dense)
    # Important: Setup BP matrices
    tensor_sim.D_inv = sim_base.D_inv
    tensor_sim.R_sparse = sim_base.R_sparse
    
    tensor_sim.x = sim_base.x.copy()
    tensor_sim.x_target = sim_base.x_target.copy()
    
    # Run with Belief Propagation (bp_steps=10 simulates local negotiation speed)
    tensor_sim.run(steps=STEPS, dt=0.05, bp_steps=10)
    
    # 4. Visualization
    plt.style.use('bmh')
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10))
    
    # Error Convergence
    ax1.plot(market_sim.history['error'], 'r--', linewidth=1.5, label='Market (Scalar / Local)')
    ax1.plot(tensor_sim.history['error'], 'g-', linewidth=2.0, label='Tensor (BP / Synergy)')
    ax1.set_yscale('log')
    ax1.set_ylabel('Resource Mismatch Norm')
    ax1.set_title(f'Strict Dynamics on Bitcoin Alpha Trust Network (N={N})')
    ax1.legend()
    
    # Entropy Production Rate
    ax2.plot(market_sim.history['entropy_production'], 'r--', linewidth=1.5, label='Market Dissipation')
    ax2.plot(tensor_sim.history['entropy_production'], 'g-', linewidth=2.0, label='Tensor Dissipation')
    ax2.set_ylabel('Entropy Production Rate (d$S$/dt)')
    ax2.set_xlabel('Time Steps')
    ax2.set_yscale('log')
    ax2.set_title('Thermodynamic Efficiency: Real Trust Network')
    ax2.legend()
    
    output_path = 'simulation_result_real_strict.png'
    plt.savefig(output_path, dpi=300)
    print(f"Strict analysis results saved to {output_path}")

if __name__ == "__main__":
    run_real_world_strict_analysis()

