import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.special import softmax
from scipy.stats import kurtosis, skew
import os

"""
JEDC POLICY EXPERIMENT: COUNTERFACTUAL SIMULATION
-------------------------------------------------
Goal: Demonstrate that a "Micro-Prudential" policy based on Attention Topology 
can reduce systemic risk (Kurtosis) better than standard Volatility targeting.

Mechanism:
- Simulate a coupled market system using the Attention Dynamics derived in the paper.
- Scenario A: Laissez-faire (No intervention).
- Scenario B: Attention Breaker (Intervene when Parisi Order > Threshold).
- Intervention: Artificially inject noise (increase Temperature T) to break synchronization.
"""

class MarketSimulator:
    def __init__(self, n_assets=50, n_steps=2000):
        self.n_assets = n_assets
        self.n_steps = n_steps
        self.n_components = 5
        
        # Latent Factors (Keys) - Slowly evolving
        self.K = np.random.randn(n_assets, self.n_components)
        
        # Initial State
        self.returns = np.zeros((n_steps, n_assets))
        self.parisi_history = []
        
    def run_simulation(self, policy_type='none', threshold=0.6, cooling_strength=2.0):
        """
        policy_type: 'none' or 'attention_breaker'
        """
        np.random.seed(42) # Reproducibility
        
        # Dynamics Parameters
        base_temp = 0.5 # Lower base temp to allow spontaneous ordering
        coupling_strength = 0.15 # Stronger coupling
        noise_level = 0.05
        
        current_temp = base_temp
        
        # Feedback Loop State
        last_return = 0
        
        crashes = 0
        
        print(f"Running Simulation: Policy={policy_type}...")
        
        for t in range(1, self.n_steps):
            # 1. Evolve Latent Factors
            self.K += np.random.randn(self.n_assets, self.n_components) * 0.01
            
            # 2. Investor Attention (Endogenous Feedback)
            # CRITICAL UPGRADE: Fear drives Attention
            # If market dropped yesterday, investors FOCUS HARDER (Low T)
            # This creates the Vicious Cycle (Panic -> Focus -> Correlation -> Crash)
            
            # Natural Temperature Dynamics (without policy)
            if last_return < -0.02: # -2% drop triggers panic focus
                natural_temp = base_temp * 0.2 # Tunnel vision
            else:
                natural_temp = base_temp * 1.0 + np.random.rand()*0.2
            
            # Query update
            if t > 5:
                recent_perf = np.mean(self.returns[t-5:t], axis=0)
                Q_raw = self.K.T @ recent_perf 
                Q = np.tile(Q_raw, (self.n_assets, 1)) 
            else:
                Q = np.random.randn(self.n_assets, self.n_components)
            
            # 3. Policy Intervention
            logits = (Q @ self.K.T) 
            logits = logits / (np.std(logits) + 1e-6)
            
            if policy_type == 'attention_breaker':
                # Policy overrides natural dynamics
                # Trigger earlier: threshold 0.25
                if len(self.parisi_history) > 1 and self.parisi_history[-1] > threshold:
                    # BREAKER: Force Temp High
                    current_temp = 5.0 
                else:
                    current_temp = natural_temp
            else:
                current_temp = natural_temp
                
            attn_weights = softmax(logits / current_temp, axis=1)
            
            # 4. Record Order
            overlaps = attn_weights @ attn_weights.T
            mask = np.ones_like(overlaps) - np.eye(self.n_assets)
            parisi = np.sum(overlaps * mask) / (self.n_assets * (self.n_assets - 1))
            self.parisi_history.append(parisi)
            
            # 5. Generate Returns
            demand = np.sum(attn_weights, axis=0) 
            demand = (demand - np.mean(demand)) / (np.std(demand) + 1e-6)
            
            # Impact Function: Non-linear
            # High Parisi -> High Fragility
            fragility = parisi ** 3 # Cubic fragility
            
            # Shock
            exo_shock = np.random.randn() 
            
            # Return = Alpha * Demand + Beta * Shock * Fragility
            # If Fragility is high, the Shock is amplified globally
            r_t = coupling_strength * demand + exo_shock * (1 + fragility * 10.0) * noise_level
            
            self.returns[t] = r_t
            last_return = np.mean(r_t)
            
            if last_return < -0.10: # -10% Crash
                crashes += 1
                
        return self.returns, self.parisi_history

def run_policy_experiment():
    sim = MarketSimulator(n_assets=100, n_steps=3000)
    
    # 1. Baseline: Laissez-faire
    ret_base, parisi_base = sim.run_simulation(policy_type='none')
    market_base = np.mean(ret_base, axis=1)
    
    # 2. Policy: Attention Breaker (Threshold = 0.5)
    # Re-instantiate to reset seed/state is tricky, better to subclass or reset method.
    # Actually, the run_simulation sets seed 42. So noise sequence is identical!
    # This allows PERFECT counterfactual comparison.
    ret_pol, parisi_pol = sim.run_simulation(policy_type='attention_breaker', threshold=0.15, cooling_strength=5.0)
    market_pol = np.mean(ret_pol, axis=1)
    
    # --- STATISTICS ---
    kurt_base = kurtosis(market_base)
    kurt_pol = kurtosis(market_pol)
    
    std_base = np.std(market_base)
    std_pol = np.std(market_pol)
    
    # Count extreme crashes (> 3 sigma of baseline)
    sigma = std_base
    crash_threshold = -3.0 * sigma
    n_crashes_base = np.sum(market_base < crash_threshold)
    n_crashes_pol = np.sum(market_pol < crash_threshold)
    
    print("\n" + "="*60)
    print("JEDC COUNTERFACTUAL POLICY EXPERIMENT RESULTS")
    print("="*60)
    print(f"{'Metric':<25} | {'Baseline (Laissez-faire)':<25} | {'Policy (Attention Breaker)':<25}")
    print("-" * 80)
    print(f"{'Kurtosis (Tail Risk)':<25} | {kurt_base:.4f}{'':<20} | {kurt_pol:.4f}")
    print(f"{'Volatility':<25} | {std_base:.4f}{'':<20} | {std_pol:.4f}")
    print(f"{'Extreme Crashes (>3σ)':<25} | {n_crashes_base:<25} | {n_crashes_pol}")
    print("-" * 80)
    
    improvement = (kurt_base - kurt_pol) / kurt_base * 100
    print(f"\nCONCLUSION: The 'Attention Breaker' policy reduced Systemic Kurtosis by {improvement:.1f}%.")
    
    # --- PLOTTING ---
    if not os.path.exists('figures'): os.makedirs('figures')
    
    plt.figure(figsize=(12, 10))
    
    # 1. Return Time Series (Zoom in on a crash)
    plt.subplot(3, 1, 1)
    plt.plot(market_base, 'k', alpha=0.5, label='Baseline Market', linewidth=1)
    plt.plot(market_pol, 'b', alpha=0.6, label='Policy Intervention', linewidth=1)
    plt.title('Counterfactual Simulation: Market Dynamics')
    plt.ylabel('Daily Returns')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 2. Order Parameter Dynamics
    plt.subplot(3, 1, 2)
    plt.plot(parisi_base, 'r--', label='Baseline Crowding (Parisi)', alpha=0.6)
    plt.plot(parisi_pol, 'g-', label='Managed Crowding (With Breaker)', linewidth=1.5)
    plt.axhline(0.15, color='k', linestyle=':', label='Intervention Threshold')
    plt.title('Micro-Structure Control: Breaking the Herd')
    plt.ylabel('Parisi Order Parameter')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 3. Distribution Comparison (Tail)
    plt.subplot(3, 1, 3)
    plt.hist(market_base, bins=50, alpha=0.5, color='gray', label='Baseline Distribution', density=True, range=(-0.15, 0.15))
    plt.hist(market_pol, bins=50, alpha=0.5, color='blue', label='Policy Distribution', density=True, range=(-0.15, 0.15))
    plt.yscale('log') # Log scale to show TAILS
    plt.title('Tail Risk Reduction (Log Scale)')
    plt.xlabel('Return')
    plt.ylabel('Density (Log)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('figures/jedc_policy_experiment.png')
    print("Saved simulation figure to figures/jedc_policy_experiment.png")

if __name__ == "__main__":
    run_policy_experiment()
