import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

"""
Figure 6 Generator: Market Topology Collapse
--------------------------------------------
Visualizes the correlation matrix (Attention Topology) of the market 
in two distinct phases:
1. Stable Phase (Modular, High Entropy)
2. Crisis Phase (Mean Field, Low Entropy)
"""

def generate_topology_comparison():
    tickers = [
        "MSFT", "AMZN", "JPM", "XOM", "GE", 
        "JNJ", "PFE", "C", "KO", "PEP",
        "WMT", "MRK", "INTC", "CSCO", "IBM",
        "GS", "BAC", "AIG", "AXP", "MCD" 
    ] # Mix of Tech, Finance, Consumer
    
    print("Downloading data for Topology Visualization...")
    # Crisis: 2008-09 to 2008-12
    # Stable: 2006-01 to 2006-06
    
    start_date = "2006-01-01"
    end_date = "2009-01-01"
    
    try:
        data = yf.download(tickers, start=start_date, end=end_date)['Close']
        returns = np.log(data / data.shift(1)).dropna()
    except:
        print("Data download failed.")
        return

    # Define periods
    stable_period = returns.loc["2006-01-01":"2006-06-01"]
    crisis_period = returns.loc["2008-09-15":"2008-12-15"] # Lehman collapse
    
    # Compute Correlations
    corr_stable = stable_period.corr()
    corr_crisis = crisis_period.corr()
    
    # Clustering for better visualization (Cluster by Stable structure)
    # We want to keep the same node order to show the collapse
    clustermap = sns.clustermap(corr_stable, method='ward')
    reordered_index = clustermap.dendrogram_row.reordered_ind
    plt.close() # Don't show this plot
    
    tickers_ordered = [tickers[i] for i in reordered_index]
    
    # Reorder matrices
    corr_stable = corr_stable.iloc[reordered_index, reordered_index]
    corr_crisis = corr_crisis.iloc[reordered_index, reordered_index]
    
    # Plotting
    os.makedirs('figures', exist_ok=True)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
    
    sns.heatmap(corr_stable, ax=ax1, cmap='coolwarm', vmin=-0.5, vmax=1.0, square=True, cbar=False)
    ax1.set_title("Stable Phase (2006): Modular Geometry", fontsize=14)
    ax1.set_xlabel("")
    ax1.set_ylabel("")
    ax1.set_xticks([])
    ax1.set_yticks([])
    
    sns.heatmap(corr_crisis, ax=ax2, cmap='coolwarm', vmin=-0.5, vmax=1.0, square=True)
    ax2.set_title("Crisis Phase (2008): Topology Collapse", fontsize=14)
    ax2.set_xlabel("")
    ax2.set_ylabel("")
    ax2.set_xticks([])
    ax2.set_yticks([])
    
    plt.tight_layout()
    plt.savefig('figures/market_topology_collapse.png', dpi=300)
    print("Saved Figure 6 to figures/market_topology_collapse.png")

if __name__ == "__main__":
    generate_topology_comparison()
