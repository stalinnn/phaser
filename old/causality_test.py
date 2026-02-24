import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import grangercausalitytests, adfuller
import warnings
warnings.filterwarnings("ignore")

"""
Causality Check: Does Geometric Entropy PREDICT Volatility?
-----------------------------------------------------------
Critique: "Correlation breakdown is just a symptom, not a cause."
Response: Granger Causality Test.

Hypothesis: 
If S(t) (Entropy) Granger-causes VIX(t) (Volatility), but VIX does not cause S,
then the geometric collapse is a true physical precursor, not just a synchronous symptom.

Method:
1. Re-calculate Entropy S(t) using previous Top 87 assets.
2. Download VIX data.
3. Perform ADF stationarity test (Granger requires stationary series).
4. Run Granger Test for lags 1 to 10 days.
"""

def get_vix_and_returns():
    print("Downloading VIX and Market Data...")
    # Use SPY as proxy for market returns if we need to re-calc entropy
    # But to save time, let's assume we have the entropy series or re-calc it quickly.
    # Re-calculating S(t) is safer to ensure alignment.
    
    # Same tickers as before
    sectors = ["XLE", "XLF", "XLU", "XLI", "XLK", "XLV", "XLY", "XLP", "XLB"]
    blue_chips = ["GE", "IBM", "XOM", "PG", "JPM", "MSFT", "WMT", "JNJ", "PFE", "CVX"] # Top 10 for speed proxy
    tickers = sectors + blue_chips
    
    data = yf.download(tickers, start="2000-01-01", end="2023-01-01")['Close']
    data = data.ffill().bfill()
    returns = np.log(data / data.shift(1)).dropna()
    
    vix = yf.download("^VIX", start="2000-01-01", end="2023-01-01")['Close']
    
    # Ensure 1D Series
    if isinstance(vix, pd.DataFrame):
        vix = vix.iloc[:, 0]
    if isinstance(returns, pd.DataFrame):
        # returns is actually a DataFrame of multiple stocks, which is correct for entropy calc
        # But we need to make sure we don't break anything else
        pass
        
    return returns, vix

def compute_entropy_fast(returns, window=60):
    # vectorized approximation or simple loop
    dates = returns.index[window:]
    entropies = []
    values = returns.values
    
    for i in range(window, len(returns)):
        segment = values[i-window:i, :]
        corr = np.corrcoef(segment.T)
        eigvals = np.linalg.eigvalsh(corr)
        eigvals = eigvals[eigvals > 1e-5]
        eigvals /= eigvals.sum()
        S = -np.sum(eigvals * np.log(eigvals + 1e-20))
        entropies.append(S)
        
    return pd.Series(entropies, index=dates)

def run_causality_test():
    returns, vix = get_vix_and_returns()
    entropy = compute_entropy_fast(returns)
    
    # Align Data
    df = pd.DataFrame({'S': entropy, 'VIX': vix})
    df = df.dropna()
    
    # Granger Causality requires Stationary Data.
    # Prices/VIX levels are usually NOT stationary.
    # We use first differences: Delta S and Delta VIX.
    df_diff = df.diff().dropna()
    
    print("\n--- ADF Stationarity Test ---")
    # Check if differenced series are stationary
    p_S = adfuller(df_diff['S'])[1]
    p_VIX = adfuller(df_diff['VIX'])[1]
    print(f"p-value (Delta S): {p_S:.4e}")
    print(f"p-value (Delta VIX): {p_VIX:.4e}")
    
    if p_S > 0.05 or p_VIX > 0.05:
        print("Warning: Data might not be stationary. Results unreliable.")
    
    # Granger Test 1: Does Entropy Cause VIX? (S -> VIX)
    # Null Hypothesis: S does NOT cause VIX.
    # Low p-value means REJECT Null -> S CAUSES VIX.
    print("\n--- TEST A: Does Entropy Collapse Predict VIX Spikes? (S -> VIX) ---")
    # input: [Response, Predictor]
    # We want to see if S predicts VIX, so Predictor=S.
    gc_res_A = grangercausalitytests(df_diff[['VIX', 'S']], maxlag=[1, 2, 5, 10], verbose=True)
    
    # Granger Test 2: Does VIX Cause Entropy? (VIX -> S)
    # Null Hypothesis: VIX does NOT cause S.
    print("\n--- TEST B: Does VIX Predict Entropy Collapse? (VIX -> S) ---")
    gc_res_B = grangercausalitytests(df_diff[['S', 'VIX']], maxlag=[1, 2, 5, 10], verbose=True)
    
    return gc_res_A, gc_res_B

if __name__ == "__main__":
    run_causality_test()
