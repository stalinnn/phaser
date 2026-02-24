import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.special import softmax
from sklearn.decomposition import NMF
import os

"""
MICRO-FOUNDATION: AGENT-BASED MODEL (ABM) SIMULATION
----------------------------------------------------
Objective: Demonstrate that a rise in Parisi Order Parameter is a direct consequence 
of "Herding Behavior" among heterogeneous agents.

Model: Simplified Kirman's Ant Model / Ising Model of Opinion Dynamics.
- N Agents: 500
- State s_i: +1 (Optimistic/Buy), -1 (Pessimistic/Sell)
- Interaction: J (Herding Strength), T (Noise/Liquidity)
- Dynamics: Glauber Dynamics (Heat Bath Algorithm)
"""

def simulate_market(n_agents=500, n_steps=2000):
    np.random.seed(42)
    
    # 1. Setup
    # Agents' spin state: +/- 1
    spins = np.random.choice([-1, 1], size=n_agents)
    
    # Herding Parameter J (Social Influence)
    # We will vary J over time to simulate a "Panic Cycle"
    # J starts low (Independent), rises (Bubble), peaks (Crash), then resets.
    J_trajectory = np.concatenate([
        np.linspace(0.1, 0.5, 800),   # Normal
        np.linspace(0.5, 1.5, 400),   # Bubble forming (High Herding)
        np.linspace(1.5, 2.0, 200),   # Critical State
        np.linspace(2.0, 0.1, 100),   # Crash (Liquidity vanishes -> Reset)
        np.linspace(0.1, 0.1, 500)    # Recovery
    ])
    
    prices = [100.0]
    parisi_history = []
    avg_magnetization = []
    
    # Generate Synthetic Asset Returns based on Agents
    # We simulate M=10 correlated assets to calculate Parisi
    n_assets = 10
    asset_returns_history = []
    
    print(f"Simulating {n_agents} agents over {n_steps} steps...")
    
    for t in range(n_steps):
        J = J_trajectory[t] if t < len(J_trajectory) else 0.1
        
        # 2. Update Agents (Metropolis-Hastings / Glauber)
        # Each agent looks at the average opinion (Magnetization)
        m = np.mean(spins)
        
        # Probability to flip depends on alignment with neighbors
        # Hamiltonian H = -J * s_i * m
        # P(s_i -> -s_i) = 1 / (1 + exp(Delta E / Temperature))
        # We fix Temperature = 1, vary J instead.
        
        # Vectorized update
        local_field = J * m + 0.05 * np.random.randn(n_agents) # Add idiosyncratic noise
        prob_up = 1 / (1 + np.exp(-2 * local_field))
        
        # New states
        random_draws = np.random.rand(n_agents)
        spins = np.where(random_draws < prob_up, 1, -1)
        
        avg_mag = np.mean(spins)
        avg_magnetization.append(avg_mag)
        
        # 3. Market Price Mechanism
        # Price Change ~ Excess Demand (Magnetization)
        ret = 0.01 * avg_mag + 0.01 * np.random.randn()
        prices.append(prices[-1] * (1 + ret))
        
        # 4. Generate Multi-Asset Correlation Structure
        # Assets are coupled to the "Market Sentiment" (Mag) with different betas
        betas = np.linspace(0.5, 1.5, n_assets)
        idiosyncratic = 0.02 * np.random.randn(n_assets)
        
        # R_i = Beta_i * Market_Factor + Noise
        # Market_Factor is driven by the collective spin state
        mkt_factor = avg_mag + 0.1 * np.random.randn()
        
        # Crucial: The "Noise" decreases as J increases (Herding dominates)
        # This simulates "Liquidity Black Hole"
        noise_level = 1.0 / (1 + J*2) 
        
        r_t = betas * mkt_factor + noise_level * idiosyncratic
        asset_returns_history.append(r_t)
        
    # Convert to DataFrame
    df_ret = pd.DataFrame(asset_returns_history)
    
    # 5. Calculate TFI (Topological Fragility Index) on Synthetic Data
    # Rolling Window
    window = 50
    tfi_curve = []
    
    print("Calculating TFI on Synthetic Data...")
    for t in range(window, len(df_ret)):
        w = df_ret.iloc[t-window:t]
        corr = w.corr().values
        
        # Simplified TFI for ABM (just off-diag mean of NMF-Attention)
        # Or even simpler: Softmax Attention Order
        try:
            # Quick NMF
            model = NMF(n_components=3, init='random', random_state=None, max_iter=10)
            W = model.fit_transform(corr + 1)
            H = model.components_
            K = H.T
            K_inv = np.linalg.pinv(K.T @ K + 0.1 * np.eye(3))
            Q = (corr + 1) @ K @ K_inv
            
            # Temp
            max_c = np.max(corr - np.eye(n_assets))
            T = 1.0 / (max_c + 1e-3)
            T = np.clip(T, 0.1, 2.0)
            
            Attn = softmax(Q/T, axis=1)
            overlaps = Attn @ Attn.T
            q = np.mean(overlaps[np.triu_indices(n_assets, k=1)])
            tfi_curve.append(q)
        except:
            tfi_curve.append(0)
            
    # Plotting
    if not os.path.exists('JEDC_Submission_Package1/figures'):
        os.makedirs('JEDC_Submission_Package1/figures')
        
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 12), sharex=True)
    
    # Plot J (Herding)
    time_steps = np.arange(len(J_trajectory))
    ax1.plot(time_steps, J_trajectory, 'g-', label='Interaction Strength (J)')
    ax1.set_ylabel('Herding Intensity (J)')
    ax1.set_title('A. Agent Interaction (Control Parameter)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot Price
    ax2.plot(prices, 'k-', label='Synthetic Market Price')
    ax2.set_ylabel('Price')
    ax2.set_title('B. Emergent Price Dynamics')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Plot TFI
    t_idx = np.arange(window, len(df_ret))
    ax3.plot(t_idx, tfi_curve, 'r-', linewidth=2, label='Topological Fragility Index (TFI)')
    ax3.set_ylabel('TFI (Order Parameter)')
    ax3.set_title('C. Micro-Structure Warning Signal')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Highlight Crash Zone
    crash_start = 1200
    crash_end = 1400
    ax3.axvspan(crash_start, crash_end, color='red', alpha=0.2, label='Phase Transition Zone')
    
    ax3.set_ylabel('Topological Order (q)')
    ax3.set_xlabel('Simulation Steps')
    ax3.set_title('C. Micro-Structure Warning Signal')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('JEDC_Submission_Package1/figures/abm_simulation_proof.png')
    print("ABM Simulation complete. Figure saved to figures/abm_simulation_proof.png")

if __name__ == "__main__":
    simulate_market()
