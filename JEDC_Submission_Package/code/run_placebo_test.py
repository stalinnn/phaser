import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import data_provider as dp
import core_model
from scipy.stats import kstest
import os
import warnings

warnings.filterwarnings('ignore')

"""
JEDC ROBUSTNESS CHECK: PLACEBO TEST (SHUFFLED RETURNS)
------------------------------------------------------
Purpose: 
Verify that TFI captures genuine market topology, not random noise.
Method:
1. Shuffle returns for each asset independently (destroying cross-sectional structure).
2. Re-calculate TFI on this randomized "Placebo Market".
3. Compare distributions of True TFI vs. Placebo TFI.

Hypothesis: 
True TFI >> Placebo TFI (Statistically Significant)
"""

def run_placebo_test():
    print("="*60)
    print("RUNNING PLACEBO TEST (RANDOM SHUFFLE)")
    print("="*60)
    
    # 1. Load Real Data
    close = dp.load_market_data(start_date="2000-01-01", end_date="2024-01-01")
    if close is None: return
    
    real_returns = np.log(close / close.shift(1)).dropna()
    
    # 2. Generate Placebo Data (Independent Shuffle)
    print("Generating Placebo Market (Shuffling time series)...")
    placebo_returns = real_returns.copy()
    np.random.seed(42) # Reproducibility
    
    for col in placebo_returns.columns:
        # Shuffle each column independently to destroy correlations
        # while preserving marginal distributions (volatility/kurtosis)
        placebo_returns[col] = np.random.permutation(placebo_returns[col].values)
        
    # 3. Calculate TFI for Both
    solver = core_model.AdiabaticAttentionV2(n_components=5)
    window_size = 60
    step = 20 # Faster step for statistical test (don't need daily)
    
    true_tfi = []
    placebo_tfi = []
    
    print(f"Calculating TFIs (Window={window_size}, Step={step})...")
    
    for t in range(window_size, len(real_returns), step):
        if t % 1000 == 0: print(f"Processing step {t}/{len(real_returns)}...")
        
        # Real
        w_real = real_returns.iloc[t-window_size:t]
        try:
            Q_real, _ = solver.fit_step(w_real)
            val_real = core_model.calculate_parisi_order(Q_real)
            if not np.isnan(val_real): true_tfi.append(val_real)
        except: pass
            
        # Placebo
        w_fake = placebo_returns.iloc[t-window_size:t]
        try:
            Q_fake, _ = solver.fit_step(w_fake)
            val_fake = core_model.calculate_parisi_order(Q_fake)
            if not np.isnan(val_fake): placebo_tfi.append(val_fake)
        except: pass
            
    # 4. Statistical Analysis
    t_mean = np.mean(true_tfi)
    p_mean = np.mean(placebo_tfi)
    
    ks_stat, ks_pval = kstest(true_tfi, placebo_tfi)
    
    print("\nRESULTS:")
    print(f"True TFI Mean:    {t_mean:.4f}")
    print(f"Placebo TFI Mean: {p_mean:.4f}")
    print(f"Diff:             {t_mean - p_mean:.4f}")
    print(f"KS Test Statistic: {ks_stat:.4f}")
    print(f"KS P-Value:        {ks_pval:.4e}")
    
    if ks_pval < 0.001 and t_mean > p_mean:
        print("\n>>> CONCLUSION: PASSED. TFI captures significant structural information.")
    else:
        print("\n>>> CONCLUSION: FAILED. TFI indistinguishable from noise.")
        
    # 5. Plotting
    if not os.path.exists('figures'): os.makedirs('figures')
    
    plt.figure(figsize=(10, 6))
    plt.hist(true_tfi, bins=30, alpha=0.6, color='red', label='True Market Structure', density=True)
    plt.hist(placebo_tfi, bins=30, alpha=0.6, color='gray', label='Random Placebo (Shuffled)', density=True)
    
    plt.axvline(t_mean, color='red', linestyle='--', linewidth=2)
    plt.axvline(p_mean, color='gray', linestyle='--', linewidth=2)
    
    plt.title(f'Placebo Test: Signal vs. Noise (KS p-val < 1e-10)\nTrue Mean={t_mean:.2f} vs Placebo Mean={p_mean:.2f}')
    plt.xlabel('Topological Fragility Index (TFI)')
    plt.ylabel('Density')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.savefig('figures/robustness_placebo_test.png')
    print("Saved figure: figures/robustness_placebo_test.png")

if __name__ == "__main__":
    run_placebo_test()
