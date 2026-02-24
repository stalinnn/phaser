import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

"""
Empirical Finance Analysis: Geometric Entropy & Market Crashes
------------------------------------------------------------
Objective:
Verify the paper's hypothesis that "Financial Crashes" correspond to a collapse 
in the system's "Geometric Coordination Depth" (K -> 1).

Method:
1. Download historical data for Top 20 S&P 500 stocks (2018-2022).
2. Compute the Rolling Correlation Matrix R(t).
3. Compute the Spectrum (Eigenvalues) of R(t).
4. Calculate "Von Neumann Entropy" (VNE) of the spectrum.
   - High VNE: Diverse, high-dimensional geometric structure (Healthy).
   - Low VNE: Collapse to a single dominant mode (Panic/Blind Coordination).
"""

def get_data():
    # Top 20 US Stocks by approximate market cap weight (mix of sectors)
    tickers = [
        "AAPL", "MSFT", "AMZN", "GOOGL", "META", 
        "NVDA", "TSLA", "JPM", "V", "UNH", 
        "JNJ", "BAC", "PG", "HD", "XOM", 
        "MA", "CVX", "ABBV", "KO", "PEP"
    ]
    
    print("Downloading market data...")
    # Download data from 2018 to end of 2021 (Covers 2018 correction & 2020 Covid)
    data = yf.download(tickers, start="2018-01-01", end="2022-01-01")['Close']
    
    # Fill missing values if any
    data = data.ffill().bfill()
    
    # Calculate Log Returns
    returns = np.log(data / data.shift(1)).dropna()
    return data, returns

def compute_spectral_entropy(returns, window=30):
    """
    Computes the Von Neumann Entropy of the correlation matrix over a rolling window.
    S = - sum(lambda_i * log(lambda_i)) where lambda are normalized eigenvalues.
    """
    dates = returns.index[window:]
    entropies = []
    max_eigs = []
    
    print(f"Computing spectral metrics (Window={window} days)...")
    
    for i in range(window, len(returns)):
        # Get window
        window_data = returns.iloc[i-window:i]
        
        # Compute Correlation Matrix
        corr_matrix = window_data.corr().values
        
        # Eigen decomposition
        # We use eigh because correlation matrices are symmetric
        eigvals = np.linalg.eigvalsh(corr_matrix)
        
        # Filter small numerical noise and normalize
        eigvals = eigvals[eigvals > 1e-10]
        eigvals = eigvals / np.sum(eigvals)
        
        # Von Neumann Entropy
        vne = -np.sum(eigvals * np.log(eigvals + 1e-20)) # epsilon for log(0)
        
        entropies.append(vne)
        max_eigs.append(np.max(eigvals)) # Dominant mode strength
        
    return pd.Series(entropies, index=dates), pd.Series(max_eigs, index=dates)

def plot_results(prices, entropy, max_eigs):
    # Create a proxy market index (Mean normalized price)
    market_index = prices.mean(axis=1)
    market_index = market_index / market_index.iloc[0] * 100
    
    # Align dates
    common_index = entropy.index
    market_index = market_index.loc[common_index]
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    
    # 1. Market Price
    ax1.plot(market_index.index, market_index.values, color='black', label='Market Index (Top 20 Proxy)')
    ax1.set_ylabel('Index Price (Normalized)')
    ax1.set_title('Market Dynamics vs. Geometric Entropy')
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc='upper left')
    
    # Highlight 2020 Crash
    crash_start = datetime(2020, 2, 19)
    crash_end = datetime(2020, 3, 23)
    ax1.axvspan(crash_start, crash_end, color='red', alpha=0.1, label='Covid Crash')
    
    # 2. Geometric Entropy (The "Attention" Metric)
    # We plot Entropy on left axis, Dominant Mode on right (inverted)
    
    color = 'tab:blue'
    ax2.set_ylabel('Geometric Entropy (Complexity)', color=color)
    ax2.plot(entropy.index, entropy.values, color=color, linewidth=1.5, label='Spectral Entropy')
    ax2.tick_params(axis='y', labelcolor=color)
    
    # Add simple moving average to smooth noise
    entropy_ma = entropy.rolling(window=10).mean()
    ax2.plot(entropy.index, entropy_ma.values, color='navy', linewidth=2, linestyle='--')
    
    # Annotate
    ax2.text(crash_start, entropy.min(), "Coordination Collapse", color='red', rotation=90, verticalalignment='bottom')

    ax2.set_xlabel('Date')
    ax2.grid(True, alpha=0.3)
    ax2.set_title('Geometric Phase Transition (Order Parameter)')
    
    plt.tight_layout()
    plt.savefig('figures/empirical_finance_entropy.png', dpi=300)
    print("Saved plot to figures/empirical_finance_entropy.png")

if __name__ == "__main__":
    try:
        data, returns = get_data()
        entropy, max_eigs = compute_spectral_entropy(returns, window=40)
        plot_results(data, entropy, max_eigs)
        print("Analysis Complete.")
    except Exception as e:
        print(f"Error occurred: {e}")

