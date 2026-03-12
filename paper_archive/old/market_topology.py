import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

"""
Mechanism Reveal: Market Attention Topology Collapse
----------------------------------------------------
Critique: "Entropy drop is trivial/old news."
Response: "It's not just entropy; it's a Topological Transition of the Attention Network."

Method:
1. Infer the 'Effective Attention Matrix' A(t) from the Correlation Matrix C(t).
   - In our theory, A ~ exp(beta * C). (Softmax-like relationship).
   - Or simply visualizing the raw Correlation Matrix is enough if sorted by Sector.
2. Compare two snapshots:
   - Stable Period (e.g., Mid 2007) -> Expect Modular/Band-Diagonal Structure.
   - Crisis Period (e.g., Late 2008) -> Expect Global Collapse (All-to-All).
3. This proves the mechanism: The market loses its "Geometry" (modularity) and becomes a "Mean Field".
"""

def get_sector_tickers():
    # We need to sort tickers by SECTOR to see the block-diagonal structure visually.
    # Grouping roughly by GICS sectors using major blue chips.
    
    sectors = {
        'Tech': ['MSFT', 'AAPL', 'INTC', 'CSCO', 'ORCL', 'ADBE', 'IBM', 'NVDA', 'QCOM', 'TXN'],
        'Finance': ['JPM', 'BAC', 'C', 'WFC', 'GS', 'MS', 'AXP', 'USB', 'BLK', 'SPG'],
        'Health': ['JNJ', 'PFE', 'MRK', 'UNH', 'ABT', 'LLY', 'BMY', 'AMGN', 'GILD', 'CVS'],
        'Energy': ['XOM', 'CVX', 'COP', 'SLB', 'EOG', 'OXY', 'HAL', 'MPC', 'VLO', 'KMI'],
        'Consumer': ['PG', 'KO', 'PEP', 'WMT', 'COST', 'MCD', 'DIS', 'NKE', 'HD', 'SBUX']
    }
    
    flat_list = []
    labels = []
    for sec, ticks in sectors.items():
        flat_list.extend(ticks)
        labels.extend([sec] * len(ticks))
        
    return flat_list, labels

def get_data(tickers):
    print("Downloading sorted sector data...")
    # Need data covering 2007-2009 for the contrast
    data = yf.download(tickers, start="2006-01-01", end="2010-01-01")['Close']
    
    # DROP columns that are mostly NaN (failed downloads)
    data = data.dropna(axis=1, how='all')
    # Fill remaining small holes
    data = data.ffill().bfill()
    
    returns = np.log(data / data.shift(1)).dropna()
    
    # Filter tickers that actually survived in returns
    valid_tickers = [t for t in tickers if t in returns.columns]
    returns = returns[valid_tickers] # Sort by sector again
    
    return returns

def compute_and_plot_topology(returns, labels):
    # Snapshot 1: Stable Regime (Pre-Crisis)
    # Date: Mid 2006 to Mid 2007
    start_stable = '2006-06-01'
    end_stable = '2007-06-01'
    ret_stable = returns.loc[start_stable:end_stable]
    
    # Snapshot 2: Crisis Regime (Lehman Collapse)
    # Date: Sep 2008 to Mar 2009
    start_crisis = '2008-09-01'
    end_crisis = '2009-03-01'
    ret_crisis = returns.loc[start_crisis:end_crisis]
    
    # Compute Correlations
    corr_stable = ret_stable.corr()
    corr_crisis = ret_crisis.corr()
    
    # Visualization
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    
    # Plot settings
    cmap = 'viridis' # 'coolwarm' or 'viridis'
    
    # 1. Stable
    sns.heatmap(corr_stable, ax=axes[0], cmap=cmap, vmin=0, vmax=1, 
                xticklabels=False, yticklabels=False, cbar=False)
    axes[0].set_title(f"Stable Regime ({start_stable} to {end_stable})\nModular Attention Structure", fontsize=14)
    axes[0].set_xlabel("Assets (Sorted by Sector)", fontsize=12)
    axes[0].set_ylabel("Assets (Sorted by Sector)", fontsize=12)
    
    # Add Sector Boxes (Visual Guide)
    # We know each sector has 10 stocks.
    for i in range(5):
        rect = plt.Rectangle((i*10, i*10), 10, 10, fill=False, edgecolor='white', lw=2)
        axes[0].add_patch(rect)

    # 2. Crisis
    sns.heatmap(corr_crisis, ax=axes[1], cmap=cmap, vmin=0, vmax=1, 
                xticklabels=False, yticklabels=False, cbar=True)
    axes[1].set_title(f"Crisis Regime ({start_crisis} to {end_crisis})\nTopological Collapse (Mean Field)", fontsize=14)
    axes[1].set_xlabel("Assets (Sorted by Sector)", fontsize=12)
    
    # Add Sector Boxes
    for i in range(5):
        rect = plt.Rectangle((i*10, i*10), 10, 10, fill=False, edgecolor='white', lw=2, alpha=0.5)
        axes[1].add_patch(rect)

    plt.suptitle("Mechanism of Crisis: The Topological Phase Transition of Market Attention", fontsize=16, y=0.98)
    plt.tight_layout()
    plt.savefig('figures/market_topology_collapse.png', dpi=300)
    print("Saved to figures/market_topology_collapse.png")

if __name__ == "__main__":
    tickers, labels = get_sector_tickers()
    try:
        returns = get_data(tickers)
        
        # Update valid tickers and labels to match data
        valid_tickers = [t for t in tickers if t in returns.columns]
        
        # We need to filter labels too to keep them aligned
        valid_labels = []
        for t, l in zip(tickers, labels):
            if t in returns.columns:
                valid_labels.append(l)
                
        # Now reorder returns safely
        returns = returns[valid_tickers]
        
        compute_and_plot_topology(returns, valid_labels)
    except Exception as e:
        print(f"Error: {e}")

