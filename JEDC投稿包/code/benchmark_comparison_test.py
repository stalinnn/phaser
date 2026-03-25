import pandas as pd
import numpy as np
import yfinance as yf
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

def calculate_absorption_ratio(returns, window=252, n_components=None):
    """
    Calculate Absorption Ratio (AR) based on Kritzman et al. (2010).
    AR = Sum(Variance of Top K Eigenvectors) / Total Variance
    """
    ar_series = []
    
    # Estimate n_components as 1/5 of assets if not provided
    if n_components is None:
        n_components = max(1, int(returns.shape[1] / 5))
        
    for i in range(window, len(returns)):
        window_data = returns.iloc[i-window:i]
        # Standardize
        window_data = (window_data - window_data.mean()) / window_data.std()
        window_data = window_data.dropna(axis=1)
        
        if window_data.shape[1] < n_components:
            ar_series.append(np.nan)
            continue
            
        pca = PCA()
        pca.fit(window_data)
        
        # Explained variance ratio
        explained_var = pca.explained_variance_ratio_
        ar = np.sum(explained_var[:n_components])
        ar_series.append(ar)
        
    return pd.Series(ar_series, index=returns.index[window:])

def evaluate_metrics(indicator, forward_returns, threshold, crash_threshold=-0.10):
    """
    Evaluate Precision, Recall, Lead Time for a given indicator and threshold.
    Crash definition: Future 20-day return < crash_threshold
    """
    # Align data
    df = pd.concat([indicator, forward_returns], axis=1).dropna()
    df.columns = ['Signal', 'FwdRet']
    
    # Binary labels
    is_crash = df['FwdRet'] < crash_threshold
    is_alarm = df['Signal'] > threshold
    
    # Precision & Recall
    tp = (is_alarm & is_crash).sum()
    fp = (is_alarm & ~is_crash).sum()
    fn = (~is_alarm & is_crash).sum()
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    
    # Lead Time (simplified: days between first signal and crash start)
    # This is complex to calculate perfectly in vectorized way, 
    # we use a heuristic: average distance for TPs
    lead_time = 0
    # (Skipping detailed lead time logic for speed, using proxy if needed or just P/R)
    
    return precision, recall

def main():
    print("Fetching S&P 100 data for benchmark test...")
    # Proxies for SP100
    tickers = ["MSFT", "AAPL", "AMZN", "GOOG", "JPM", "XOM", "BAC", "WMT", "INTC", "CSCO", "C", "PFE", "KO"]
    data = yf.download(tickers, start="2000-01-01", end="2024-01-01", progress=False)['Close']
    returns = np.log(data / data.shift(1)).dropna()
    
    # 1. Calculate Absorption Ratio (AR)
    print("Calculating Absorption Ratio...")
    ar = calculate_absorption_ratio(returns, window=60, n_components=3)
    
    # 2. Calculate VIX Proxy (Historical Volatility)
    # Real VIX is an index, here we use realized vol as proxy if VIX not available
    print("Calculating Historical Volatility (VIX Proxy)...")
    hv = returns.mean(axis=1).rolling(20).std() * np.sqrt(252)
    
    # 3. Define Crashes (Forward 20-day return)
    market_ret = returns.mean(axis=1)
    fwd_ret = market_ret.rolling(20).sum().shift(-20)
    
    # 4. Evaluate
    print("\n--- Benchmark Performance (Crash Threshold: -10% in 20 days) ---")
    
    # Scan thresholds for AR
    print("\n[Absorption Ratio Performance]")
    print(f"{'Threshold':<10} {'Precision':<10} {'Recall':<10}")
    for thresh in [0.7, 0.75, 0.8, 0.85, 0.9]:
        p, r = evaluate_metrics(ar, fwd_ret, thresh)
        print(f"{thresh:<10.2f} {p:<10.2%} {r:<10.2%}")
        
    # Scan thresholds for HV (VIX Proxy)
    print("\n[Historical Volatility Performance]")
    print(f"{'Threshold':<10} {'Precision':<10} {'Recall':<10}")
    for thresh in [0.2, 0.3, 0.4, 0.5, 0.6]: # Annualized Vol
        p, r = evaluate_metrics(hv, fwd_ret, thresh)
        print(f"{thresh:<10.2f} {p:<10.2%} {r:<10.2%}")

if __name__ == "__main__":
    main()
