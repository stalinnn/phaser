import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from simulation import EconomySimulationBase, SmartMarketSimulation, TensorSynergySimulation
import scipy.sparse as sp

def run_crossover_plot():
    """
    Exp 08: The Crossover Line Plot
    
    Visualizes the exact moment when Tensor Network becomes cheaper than Market
    as Communication Cost decreases.
    """
    print("=== Generating Crossover Line Plot ===")
    
    n_agents = 300
    steps = 100
    volatility = 2.0 # Moderate environment
    
    # X-axis: Communication Cost lambda
    # Focus on the crossover region seen in the heatmap (around 0.001 - 0.003)
    comm_costs = np.linspace(0.0, 0.004, 20)
    
    market_total_costs = []
    tensor_total_costs = []
    
    # Components breakdown
    market_mismatch = []
    market_comm = []
    tensor_mismatch = []
    tensor_comm = []
    
    trials = 5
    
    for i, cost_lambda in enumerate(comm_costs):
        print(f"Simulating Cost Lambda = {cost_lambda:.5f}...")
        
        m_costs = []
        t_costs = []
        
        m_mis = []
        m_com = []
        t_mis = []
        t_com = []
        
        for _ in range(trials):
            sim_base = EconomySimulationBase(n_agents=n_agents, graph_type='scale_free')
            
            # Market
            m_sim = SmartMarketSimulation(n_agents=n_agents)
            m_sim.G_dense = sim_base.G_dense.copy()
            m_sim.G_sparse = sp.csr_matrix(m_sim.G_dense)
            m_sim.x = sim_base.x.copy()
            m_sim.x_target = sim_base.x_target.copy()
            m_sim.run(steps=steps, dt=0.02, volatility=volatility, arbitrage_coeff=0.5, momentum_coeff=0.2)
            
            m_err = np.sum(m_sim.history['error'])
            m_ops = m_sim.total_comm_ops * cost_lambda
            m_costs.append(m_err + m_ops)
            m_mis.append(m_err)
            m_com.append(m_ops)
            
            # Tensor
            t_sim = TensorSynergySimulation(n_agents=n_agents)
            t_sim.G_dense = sim_base.G_dense.copy()
            t_sim.G_sparse = sp.csr_matrix(t_sim.G_dense)
            t_sim.D_inv = sim_base.D_inv
            t_sim.R_sparse = sim_base.R_sparse
            t_sim.x = sim_base.x.copy()
            t_sim.x_target = sim_base.x_target.copy()
            t_sim.run(steps=steps, dt=0.05, volatility=volatility, bp_steps=5)
            
            t_err = np.sum(t_sim.history['error'])
            t_ops = t_sim.total_comm_ops * cost_lambda
            t_costs.append(t_err + t_ops)
            t_mis.append(t_err)
            t_com.append(t_ops)
            
        market_total_costs.append(np.mean(m_costs))
        tensor_total_costs.append(np.mean(t_costs))
        
        market_mismatch.append(np.mean(m_mis))
        market_comm.append(np.mean(m_com))
        tensor_mismatch.append(np.mean(t_mis))
        tensor_comm.append(np.mean(t_com))

    # Plotting
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Plot 1: Total Cost Crossover
    ax1.plot(comm_costs, market_total_costs, 'r-o', linewidth=2, label='Market (Smart)')
    ax1.plot(comm_costs, tensor_total_costs, 'b-s', linewidth=2, label='Tensor Network')
    
    # Find crossover point
    idx = np.argwhere(np.diff(np.sign(np.array(market_total_costs) - np.array(tensor_total_costs)))).flatten()
    if len(idx) > 0:
        cross_x = comm_costs[idx[0]]
        cross_y = market_total_costs[idx[0]]
        ax1.plot(cross_x, cross_y, 'k*', markersize=20, label='Phase Transition Point')
        ax1.annotate(f'Critical Cost\n$\lambda_c \\approx {cross_x:.4f}$', 
                     xy=(cross_x, cross_y), xytext=(cross_x+0.001, cross_y+1000),
                     arrowprops=dict(facecolor='black', shrink=0.05))

    ax1.set_xlabel('Unit Communication Cost ($\lambda$)', fontsize=12)
    ax1.set_ylabel('Total Thermodynamic Cost (Free Energy)', fontsize=12)
    ax1.set_title('The Evolution of Economic Efficiency', fontsize=14)
    ax1.legend(fontsize=12)
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Component Breakdown (Stacked Area)
    # Normalize to 100% to show composition change? No, let's show raw values to see the tradeoff.
    
    # We pick 3 representative points: Low, Medium, High Cost
    indices = [0, 10, 19]
    labels = ['Free Info', 'Moderate Cost', 'High Cost']
    x = np.arange(len(indices))
    width = 0.35
    
    m_vals = [market_total_costs[i] for i in indices]
    t_vals = [tensor_total_costs[i] for i in indices]
    
    rects1 = ax2.bar(x - width/2, m_vals, width, label='Market', color='indianred')
    rects2 = ax2.bar(x + width/2, t_vals, width, label='Tensor', color='steelblue')
    
    ax2.set_ylabel('Total Cost')
    ax2.set_title('Cost Comparison at Different Regimes')
    ax2.set_xticks(x)
    ax2.set_xticklabels([f'{comm_costs[i]:.4f}' for i in indices])
    ax2.set_xlabel('Communication Cost ($\lambda$)')
    ax2.legend()
    
    plt.tight_layout()
    plt.savefig('figures/simulation_crossover_plot.png', dpi=300)
    print("Saved to figures/simulation_crossover_plot.png")

if __name__ == "__main__":
    run_crossover_plot()

