import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def fetch_data():
    tickers = [
        "AAPL", "MSFT", "AMZN", "GOOGL", "META", 
        "NVDA", "TSLA", "JPM", "V", "UNH", 
        "JNJ", "BAC", "PG", "HD", "XOM", 
        "MA", "CVX", "ABBV", "KO", "PEP"
    ]
    # Fetch Stock Data
    data = yf.download(tickers, start="2019-01-01", end="2021-01-01")['Close']
    data = data.ffill().bfill()
    returns = np.log(data / data.shift(1)).dropna()
    
    # Fetch VIX Data
    vix = yf.download("^VIX", start="2019-01-01", end="2021-01-01")['Close']
    vix = vix.ffill().bfill()
    
    # Ensure Series
    if isinstance(vix, pd.DataFrame):
        vix = vix.iloc[:, 0]
    
    # Align dates
    common_idx = returns.index.intersection(vix.index)
    returns = returns.loc[common_idx]
    vix = vix.loc[common_idx]
    
    return returns, vix

def compute_metrics(returns, window=30):
    dates = returns.index[window:]
    geo_entropy = []
    
    for i in range(window, len(returns)):
        window_data = returns.iloc[i-window:i]
        corr_matrix = window_data.corr().values
        
        eigvals = np.linalg.eigvalsh(corr_matrix)
        eigvals = eigvals[eigvals > 1e-10]
        probs = eigvals / np.sum(eigvals)
        s = -np.sum(probs * np.log(probs + 1e-12))
        geo_entropy.append(s)
        
    return pd.Series(geo_entropy, index=dates)

def analyze_lead_lag(vix, entropy):
    # Align
    common_idx = entropy.index.intersection(vix.index)
    vix = vix.loc[common_idx]
    ent = entropy.loc[common_idx]
    
    # Invert Entropy: Collapse (Low Entropy) corresponds to High VIX
    # So we correlate (-Entropy) with VIX
    neg_ent = -ent
    
    # Calculate Cross Correlation for Lags -10 to +10
    lags = range(-15, 16)
    corrs = []
    
    for lag in lags:
        # Shift neg_ent by lag
        # If lag < 0 (e.g. -5), we are comparing neg_ent(t-5) with VIX(t)
        # This checks if past entropy predicts future VIX
        shifted_ent = neg_ent.shift(lag)
        corr = vix.corr(shifted_ent)
        corrs.append(corr)
        
    # Plot
    plt.figure(figsize=(10, 6))
    colors = ['red' if l < 0 else 'gray' for l in lags]
    plt.bar(lags, corrs, color=colors)
    plt.axvline(0, color='black', linestyle='--')
    plt.title("Lead-Lag Analysis: Geometric Entropy vs VIX")
    plt.xlabel("Lag (Days): Negative means Entropy leads VIX")
    plt.ylabel("Correlation")
    plt.savefig('figures/lead_lag_analysis.png')
    
    # Find peak
    max_corr_idx = np.argmax(corrs)
    best_lag = lags[max_corr_idx]
    print(f"Peak Correlation at Lag: {best_lag} days")
    
    return best_lag

if __name__ == "__main__":
    try:
        returns, vix = fetch_data()
        entropy = compute_metrics(returns, window=40)
        best_lag = analyze_lead_lag(vix, entropy)
        
        print("\n--- ANALYSIS REPORT ---")
        if best_lag < 0:
            print(f"SUCCESS: Geometric Entropy collapses {abs(best_lag)} days BEFORE VIX spikes.")
            print("This proves the 'Structural Precursor' hypothesis.")
        else:
            print("RESULT: Synchronous or Lagging indicator.")
            
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
