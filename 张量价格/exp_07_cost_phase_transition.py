import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from simulation import EconomySimulationBase, SmartMarketSimulation, TensorSynergySimulation
import scipy.sparse as sp
import time
import os

def run_cost_phase_diagram():
    """
    Exp 07: Communication-Convergence Trade-off Analysis
    
    Total Cost = Integral(Error) + lambda * (Information Bits Flow)
    
    对比：
    1. Second-Order Network: 迭代少，单步通信大 (High Bandwidth)
    2. First-Order Network: 迭代多，单步通信小 (Low Bandwidth)
    """
    print("=== Initiating Cost-Efficiency Trade-off Analysis ===")
    
    # Parameters
    n_agents = 200 
    
    # Y-axis: Environmental Volatility (Noise)
    volatility_levels = np.linspace(0.1, 5.0, 8)
    
    # X-axis: Unit Cost of Information (lambda)
    # 之前是 ops，现在是 bits (float-hops)。数量级会大很多，所以 lambda 要小一点。
    comm_cost_levels = np.linspace(0.0, 0.001, 8) 
    
    # Grid: Positive = Tensor Wins (Lower Total Cost), Negative = Scalar Wins
    advantage_grid = np.zeros((len(volatility_levels), len(comm_cost_levels)))
    
    steps = 100 # 增加步数以体现长期收敛差异
    trials = 3
    
    total_iter = len(volatility_levels) * len(comm_cost_levels)
    curr = 0
    
    for i, vol in enumerate(volatility_levels):
        for j, cost_lambda in enumerate(comm_cost_levels):
            curr += 1
            print(f"[{curr}/{total_iter}] Volatility={vol:.1f}, InfoCost={cost_lambda:.5f}...")
            
            diffs = []
            for _ in range(trials):
                sim_base = EconomySimulationBase(n_agents=n_agents, graph_type='scale_free')
                
                # 1. Adaptive Momentum (Benchmark: Adam/Smart Scalar)
                # 这是一个极其强劲的对手，拥有“历史记忆”和“风险适应”能力
                m_sim = SmartMarketSimulation(n_agents=n_agents)
                m_sim.G_dense = sim_base.G_dense.copy()
                m_sim.G_sparse = sp.csr_matrix(m_sim.G_dense)
                m_sim.x = sim_base.x.copy()
                m_sim.x_target = sim_base.x_target.copy()
                
                # Adam 参数配置
                # dt 可以稍微大一点，因为 Adam 比较稳
                m_sim.run(steps=steps, dt=0.02, volatility=vol, 
                          beta1=0.9, beta2=0.999, diffusion_coeff=0.5)
                
                # First-Order Total Cost
                m_error_cost = np.sum(m_sim.history['error']) 
                m_info_cost = m_sim.total_information_flow * cost_lambda
                m_total = m_error_cost + m_info_cost
                
                # 2. Tensor Network (Challenger)
                # 二阶网络每步通信量大 (bp_steps=5)，但能快速锁死最优解，累积误差小
                t_sim = TensorSynergySimulation(n_agents=n_agents)
                t_sim.G_dense = sim_base.G_dense.copy()
                t_sim.G_sparse = sp.csr_matrix(t_sim.G_dense)
                t_sim.D_inv = sim_base.D_inv
                t_sim.R_sparse = sim_base.R_sparse
                t_sim.x = sim_base.x.copy()
                t_sim.x_target = sim_base.x_target.copy()
                # 二阶网络允许更大的 dt，因为方向准
                t_sim.run(steps=steps, dt=0.04, volatility=vol, bp_steps=5) 
                
                # Second-Order Total Cost
                t_error_cost = np.sum(t_sim.history['error'])
                t_info_cost = t_sim.total_information_flow * cost_lambda
                t_total = t_error_cost + t_info_cost
                
                # Relative Advantage
                # If First-Order cost is 100, Second-Order cost is 80, result is 0.2 (20% improvement)
                diffs.append((m_total - t_total) / (m_total + 1e-9))
                
            advantage_grid[i, j] = np.mean(diffs)

    # Plotting
    plt.figure(figsize=(10, 8))
    sns.set_style("whitegrid")
    
    ax = sns.heatmap(advantage_grid, 
                     xticklabels=np.round(comm_cost_levels, 5), 
                     yticklabels=np.round(volatility_levels, 1),
                     cmap="RdBu", center=0, annot=True, fmt=".2f",
                     cbar_kws={'label': 'Second-Order Advantage (Pos) vs First-Order Advantage (Neg)'})
    
    ax.invert_yaxis()
    plt.title('Phase Transition: Bandwidth vs. Convergence\n(Second-Order wins in High Volatility, First-Order wins in High Info Cost)', fontsize=14)
    plt.xlabel('Unit Cost of Information Bit ($\lambda$)', fontsize=12)
    plt.ylabel('Environmental Volatility', fontsize=12)
    
    if not os.path.exists('figures'): os.makedirs('figures')
    plt.savefig('figures/simulation_phase_diagram_cost.png', dpi=300)
    print("Saved to figures/simulation_phase_diagram_cost.png")

if __name__ == "__main__":
    run_cost_phase_diagram()
