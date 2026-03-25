import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import data_provider as dp
import warnings
import os

warnings.filterwarnings('ignore')

"""
JEDC BENCHMARKING: ABSORPTION RATIO VS. PARISI ORDER
----------------------------------------------------
Reference: Kritzman, M., Li, Y., Page, S., & Rigobon, R. (2010). 
Principal components as a measure of systemic risk. 
Journal of Portfolio Management.

Purpose: 
Demonstrate that Parisi Order provides incremental information beyond 
the standard Absorption Ratio (which is based on linear PCA).
"""

def calculate_absorption_ratio(returns_window, n_components_fraction=0.2):
    """
    Calculate Absorption Ratio (AR) for a given window.
    AR = Sum(Variance of Top N Eigenvectors) / Total Variance
    """
    # Handle NaNs: Drop columns with any NaNs in this window
    clean_window = returns_window.dropna(axis=1)
    
    if clean_window.shape[1] < 10: return np.nan
    
    # Standardize (Correlation Matrix approach)
    # Kritzman suggests using Covariance for asset allocation, but Correlation for systemic risk
    # to avoid volatility bias. We use Correlation here.
    corr_mat = clean_window.corr().values
    
    try:
        # Eigenvalue Decomposition
        eigvals = np.linalg.eigvalsh(corr_mat)
        # Sort descending
        eigvals = np.sort(eigvals)[::-1]
        
        # Determine N (e.g., top 20% of components)
        n_comp = max(1, int(len(eigvals) * n_components_fraction))
        
        numerator = np.sum(eigvals[:n_comp])
        denominator = np.sum(eigvals)
        
        return numerator / denominator
    except:
        return np.nan

def run_benchmarking():
    # 1. Load Data
    print("Loading Data for Benchmarking...")
    close = dp.load_market_data(start_date="2000-01-01", end_date="2024-01-01")
    
    if close is None: return

    returns = np.log(close / close.shift(1))
    
    # 2. Parameters
    window_size = 252 # 1-year rolling window (Standard in literature)
    step = 5
    
    results = {}
    
    print("Calculating Absorption Ratio (Linear Benchmark)...")
    
    for t in range(window_size, len(returns), step):
        date = returns.index[t]
        window = returns.iloc[t-window_size:t]
        
        ar = calculate_absorption_ratio(window, n_components_fraction=0.2) # Top 20% eigenvalues
        
        # Simple proxy for Parisi (using Avg Corr here to save compute time for this specific check, 
        # or we could import the full model if needed. For now, let's use AvgCorr as a proxy for
        # the "Linear" part, but we really want to compare with saved Parisi data if available.
        # Ideally, we should load the Parisi results from previous runs.)
        
        results[date] = ar
        
    ar_series = pd.Series(results).sort_index()
    
    # 3. Load Previous Parisi Results (if available) or Run simple comparison
    # For this script, we'll save AR to a file so it can be plotted against Parisi later
    # Or we can plot AR vs VIX here.
    
    output_path = 'results/benchmark_absorption_ratio.csv'
    if not os.path.exists('results'): os.makedirs('results')
    
    ar_series.to_csv(output_path)
    print(f"Absorption Ratio saved to {output_path}")
    
    # 4. Plotting
    if not os.path.exists('figures'): os.makedirs('figures')
    
    plt.figure(figsize=(12, 6))
    plt.plot(ar_series.index, ar_series.values, 'k-', label='Absorption Ratio (Kritzman 2010)')
    plt.title('Systemic Risk Benchmark: Absorption Ratio (S&P 100)')
    plt.ylabel('Variance Absorbed by Top 20% PC')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.savefig('figures/benchmark_absorption_ratio.png')
    print("Saved figure: figures/benchmark_absorption_ratio.png")
    
    # 5. Interpretation Text
    print("\nINTERPRETATION FOR PAPER:")
    print("Absorption Ratio captures LINEAR mode coupling.")
    print("If Parisi Order spikes while AR is flat (or vice versa), it proves orthogonality.")
    print("Typically, AR is slow-moving, while Attention Topology can snap (Phase Transition).")

if __name__ == "__main__":
    run_benchmarking()
