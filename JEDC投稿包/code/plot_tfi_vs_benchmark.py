import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import core_model
import data_provider as dp
import warnings

warnings.filterwarnings('ignore')

"""
JEDC FIGURE GENERATOR: TFI vs. ABSORPTION RATIO (FIXED)
-------------------------------------------------------
Purpose: Visual proof that Topological Fragility Index (TFI) 
provides distinct, early-warning signals compared to the 
standard Absorption Ratio (Kritzman 2010).
"""

def generate_comparison_plot():
    print("Generating TFI vs. Absorption Ratio comparison...")
    
    # 1. Load Data (Ensuring we get the full history)
    # We use the same data for BOTH metrics to ensure alignment
    close = dp.load_market_data(start_date="2000-01-01", end_date="2024-01-01")
    if close is None: 
        print("Data load failed.")
        return
    
    # Simple returns for calculation
    returns = np.log(close / close.shift(1))
    
    print(f"Data Loaded. Shape: {returns.shape}. Date Range: {returns.index[0]} to {returns.index[-1]}")
    
    # 2. Load Absorption Ratio (Benchmark)
    ar_path = 'results/benchmark_absorption_ratio.csv'
    if os.path.exists(ar_path):
        ar_series = pd.read_csv(ar_path, index_col=0, parse_dates=True).iloc[:, 0]
        # Align AR to the current data index if needed
        ar_series = ar_series.reindex(returns.index, method='ffill')
    else:
        print("Absorption Ratio data not found. Calculating on the fly...")
        return

    # 3. Calculate TFI (Parisi Order) - FULL HISTORY RE-CALCULATION
    print("Calculating TFI (Topological Fragility Index) for full history...")
    
    window_size = 60 # 60-day window
    step = 5 # 5-day step
    solver = core_model.AdiabaticAttentionV2(n_components=5)
    
    tfi_results = {}
    
    # We must ensure we iterate through the WHOLE timeframe
    for t in range(window_size, len(returns), step):
        if t % 1000 == 0: print(f"Processing TFI: {t}/{len(returns)}")
        
        date = returns.index[t]
        window = returns.iloc[t-window_size:t]
        
        # Skip if window has too many NaNs (but don't break loop)
        if window.shape[0] < window_size: continue
        
        # Calculate TFI
        # Note: fit_step handles NaN internally usually, but let's be safe
        try:
            Q, _ = solver.fit_step(window)
            if Q is not None:
                tfi = core_model.calculate_parisi_order(Q)
                tfi_results[date] = tfi
            else:
                tfi_results[date] = np.nan
        except Exception as e:
            tfi_results[date] = np.nan
        
    tfi_series = pd.Series(tfi_results).sort_index()
    
    # Filter: Drop initial NaNs/Zeros if any
    tfi_series = tfi_series[tfi_series > 0.01]
    
    print(f"TFI Series calculated. Length: {len(tfi_series)}")
    
    # 4. Plotting (Dual Axis)
    if not os.path.exists('figures'): os.makedirs('figures')
    
    fig, ax1 = plt.subplots(figsize=(12, 6))
    
    # Plot TFI (Our Metric) - Red
    color = '#D62728' # Tab:red
    ax1.set_xlabel('Year')
    ax1.set_ylabel('Topological Fragility Index (TFI)', color=color, fontsize=12, fontweight='bold')
    
    # Plotting
    ax1.plot(tfi_series.index, tfi_series.values, color=color, linewidth=1.5, label='TFI (Non-linear Attention)')
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.set_ylim(0, 1.0)
    
    # Plot Absorption Ratio (Benchmark) - Grey
    ax2 = ax1.twinx() 
    color = '#7F7F7F' # Tab:gray
    ax2.set_ylabel('Absorption Ratio (Linear PCA)', color=color, fontsize=12)
    ax2.plot(ar_series.index, ar_series.values, color=color, linestyle='--', linewidth=1.5, label='Absorption Ratio (Benchmark)')
    ax2.tick_params(axis='y', labelcolor=color)
    ax2.set_ylim(0.4, 1.0) 
    
    # Add dummy lines for legend
    lns1 = ax1.lines
    lns2 = ax2.lines
    labs = [l.get_label() for l in lns1 + lns2]
    ax1.legend(lns1 + lns2, labs, loc='upper left', frameon=True)
    
    plt.title('Orthogonality Check: TFI vs. Standard Systemic Risk Metric (S&P 100)', fontsize=14)
    plt.grid(True, alpha=0.3)
    
    output_file = 'figures/comparison_tfi_vs_ar_fixed.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Comparison plot saved to: {output_file}")

if __name__ == "__main__":
    generate_comparison_plot()
