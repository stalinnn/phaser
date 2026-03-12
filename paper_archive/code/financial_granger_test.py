import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import NMF
from statsmodels.tsa.stattools import grangercausalitytests, adfuller
import os
import warnings
warnings.filterwarnings("ignore")

"""
SURGERY 3: GRANGER CAUSALITY TEST
---------------------------------
Goal: Prove statistically that "Query Homogeneity" (Attention Collapse) 
GRANGER-CAUSES Market Volatility (VIX/Realized Vol).

Hypothesis: H0: Query Homogeneity does NOT Granger-cause VIX.
We want to REJECT H0 with p < 0.05.
"""

def get_data_and_metric():
    # 1. Fetch Market Data for Query Homogeneity
    # Expanded Ticker List (Matching Tier 1 Analysis)
    tickers = [
        "MSFT", "AMZN", "JPM", "XOM", "GE", "JNJ", "PFE", "C", "KO", "PEP",
        "WMT", "MRK", "INTC", "CSCO", "IBM", "GS", "BAC", "AIG", "AXP", "MCD",
        "AAPL", "GOOGL", "NVDA", "TSLA", "META", "BRK-B", "UNH", "LLY", "V", "PG",
        "HD", "MA", "CVX", "ABBV", "MRK", "AVGO", "COST", "ADBE", "TMO", "PEP",
        "CSCO", "ACN", "NFLX", "LIN", "MCD", "AMD", "ABT", "DHR", "DIS", "NKE",
        "VZ", "T", "CMCSA", "TXN", "NEE", "PM", "UPS", "UNP", "BMY", "RTX",
        "HON", "LOW", "QCOM", "AMGN", "CAT", "SPGI", "MS", "BA", "INTU", "DE",
        "PLD", "BLK", "SYK", "BKNG", "GILD", "ADP", "TJX", "MDLZ", "ADI", "MMC",
        "LMT", "CVS", "CI", "VRTX", "MU", "SCHW", "REGN", "ISRG", "LRCX", "ZTS",
        "FI", "SO", "PGR", "SLB", "BDX", "BSX", "KLAC", "EOG", "PANW"
    ]
    tickers = list(set(tickers))

    start_date = "2000-01-01"
    end_date = "2024-01-01"
    
    print(f"Downloading data for Granger Test ({len(tickers)} tickers, 2000-2024)...")
    # Auto adjust ensures we get split-adjusted prices which is critical for long term
    data = yf.download(tickers, start=start_date, end=end_date, progress=True, auto_adjust=True)
    
    # Handle MultiIndex if present
    if isinstance(data.columns, pd.MultiIndex):
        if 'Close' in data.columns.get_level_values(0):
            data = data['Close']
        else:
            # Sometimes yf returns just the tickers if auto_adjust=True and only Close is relevant?
            # Or it might be (Ticker, PriceType). Let's check structure dynamically.
            # Usually with multiple tickers it is (PriceType, Ticker).
            # If we asked for just 'Close' via some param it would be different, but yf.download returns all.
            # Let's try to access 'Close' level.
            try:
                data = data['Close']
            except KeyError:
                pass # Already single level?

    # Clean data (require 80% history presence to avoid sparse matrix issues at the start)
    threshold = int(len(data) * 0.8)
    data = data.dropna(axis=1, thresh=threshold)
    data = data.ffill().dropna()
    
    returns = np.log(data / data.shift(1)).dropna()
    
    # 2. Fetch VIX (The "Crisis" variable)
    vix = yf.download("^VIX", start=start_date, end=end_date, progress=True, auto_adjust=True)
    # VIX might also be MultiIndex
    if isinstance(vix.columns, pd.MultiIndex):
         try:
            vix = vix['Close']
         except:
            pass
            
    if isinstance(vix, pd.DataFrame):
        # If still DataFrame (single col), squeeze
        if 'Close' in vix.columns:
            vix = vix['Close']
        vix = vix.squeeze()
    
    # 3. Compute Query Homogeneity (The "Predictor")
    # (Reusing logic from Surgery 2)
    # Define a stable training period for the Key matrix (e.g. 2004-2006, pre-crisis)
    stable_start = "2004-01-01"
    stable_end = "2006-01-01"
    
    stable_returns = returns.loc[stable_start:stable_end]
    corr_stable = stable_returns.corr().values
    corr_stable_pos = np.clip(corr_stable, 0, 1)
    
    # Extract Static Keys
    d_model = 3
    nmf = NMF(n_components=d_model, init='random', random_state=42)
    W_stable = nmf.fit_transform(corr_stable_pos)
    H_stable = nmf.components_
    K_static = H_stable.T
    
    # Compute Time Series of Homogeneity
    dates = []
    homogeneity = []
    
    window = 60
    step = 1 # Daily step for Granger Test (need high res)
    
    # Helper for Inverse Solve
    def inverse_solve_q(A_target, K_fixed):
        N, d = K_fixed.shape
        A_pos = np.clip(A_target, 0, None) 
        row_sums = A_pos.sum(axis=1, keepdims=True) + 1e-9
        A_prob = A_pos / row_sums
        epsilon = 1e-9
        L = np.log(A_prob + epsilon)
        L_centered = L - L.mean(axis=1, keepdims=True)
        lambda_reg = 0.1
        K_inv = np.linalg.inv(K_fixed.T @ K_fixed + lambda_reg * np.eye(d))
        Q_est = L_centered @ K_fixed @ K_inv
        return Q_est

    print("Computing Query Homogeneity Time Series...")
    for t in range(window, len(returns), step):
        current_date = returns.index[t]
        window_returns = returns.iloc[t-window:t]
        corr_t = window_returns.corr().values
        Q_t = inverse_solve_q(corr_t, K_static)
        
        # Metric
        norm_Q = np.linalg.norm(Q_t, axis=1, keepdims=True)
        Q_normalized = Q_t / (norm_Q + 1e-9)
        cosine_sim = np.dot(Q_normalized, Q_normalized.T)
        avg_sim = np.mean(cosine_sim[np.triu_indices(len(Q_t), k=1)])
        
        homogeneity.append(avg_sim)
        dates.append(current_date)
        
    ts_homogeneity = pd.Series(homogeneity, index=dates, name='Query_Homogeneity')
    
    # Align Data
    df = pd.concat([ts_homogeneity, vix], axis=1).dropna()
    df.columns = ['Homogeneity', 'VIX']
    
    return df

def run_tests():
    df = get_data_and_metric()
    
    # 1. Stationarity Check (ADF Test)
    print("\n--- Stationarity Check (ADF) ---")
    # Granger requires stationary series. We likely need to difference.
    for col in df.columns:
        res = adfuller(df[col])
        print(f"{col}: p-value = {res[1]:.4f}")
        
    # If not stationary (p > 0.05), we difference
    df_diff = df.diff().dropna()
    print("\n--- After Differencing ---")
    for col in df_diff.columns:
        res = adfuller(df_diff[col])
        print(f"{col}: p-value = {res[1]:.4f}")
        
    # 2. Granger Causality: Does Homogeneity predict VIX?
    print("\n--- Granger Causality Test (Homogeneity -> VIX) ---")
    max_lag = 10
    # The method takes a 2D array [response, predictor]
    # We want to know if Predictor (col 1) causes Response (col 0)
    # So we pass [VIX, Homogeneity]
    test_result = grangercausalitytests(df_diff[['VIX', 'Homogeneity']], maxlag=max_lag, verbose=True)
    
    # 3. Reverse Test: Does VIX predict Homogeneity? (Check for feedback loop)
    print("\n--- Reverse Granger Causality Test (VIX -> Homogeneity) ---")
    test_result_rev = grangercausalitytests(df_diff[['Homogeneity', 'VIX']], maxlag=max_lag, verbose=False)
    
    # Visualize
    # Just show the F-test p-values for forward vs reverse
    p_values_fwd = [test_result[i][0]['ssr_chi2test'][1] for i in range(1, max_lag+1)]
    p_values_rev = [test_result_rev[i][0]['ssr_chi2test'][1] for i in range(1, max_lag+1)]
    
    plt.figure(figsize=(10, 6))
    lags = range(1, max_lag+1)
    plt.plot(lags, p_values_fwd, 'b-o', label='Homogeneity -> VIX (Predictive)')
    plt.plot(lags, p_values_rev, 'r--s', label='VIX -> Homogeneity (Reactive)')
    plt.axhline(0.05, color='k', linestyle=':', label='Significance Threshold (p=0.05)')
    plt.yscale('log')
    plt.xlabel('Lag (Days)')
    plt.ylabel('p-value (Log Scale)')
    plt.title('Granger Causality Test: Query Collapse Predicts Volatility')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    os.makedirs('figures', exist_ok=True)
    plt.savefig('figures/granger_causality_test.png', dpi=300)
    print("Saved Granger Test plot to figures/granger_causality_test.png")

if __name__ == "__main__":
    run_tests()
