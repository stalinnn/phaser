import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
import yfinance as yf
import os

"""
Experiment: Is Geometric Entropy just "Mean Correlation" in disguise?
-------------------------------------------------------------------
Critique: Econophysics 101 says Entropy ~ -Mean_Correlation.
Response: Check if Entropy provides INCREMENTAL information beyond Mean Corr and VIX.

Method:
1. Compute Rolling Mean Correlation (rho_bar).
2. Compute Rolling Geometric Entropy (S).
3. Check correlation Corr(S, rho_bar).
4. Lead-Lag analysis: Does S predict VIX better than rho_bar?
"""

def analyze_incremental_value():
    # Reduced Ticker List (High reliability)
    tickers = ["AAPL", "MSFT", "AMZN", "GOOGL", "JPM", "XOM", "SPY"]
    
    print("Downloading Data (Robust Mode)...")
    data = yf.download(tickers, start="2018-01-01", end="2022-01-01")['Close']
    
    # Drop columns with all NaNs
    data = data.dropna(axis=1, how='all')
    # Fill remaining NaNs
    data = data.ffill().bfill()
    
    returns = np.log(data / data.shift(1)).dropna()
    
    # Download VIX
    vix = yf.download("^VIX", start="2018-01-01", end="2022-01-01")['Close']
    vix = vix.ffill().bfill()
    
    # Align Data
    # Ensure returns and VIX share the same index
    common_index = returns.index.intersection(vix.index)
    returns = returns.loc[common_index]
    vix = vix.loc[common_index]
    
    if len(returns) < 50:
        print("Error: Not enough data downloaded.")
        return

    window = 30
    dates = returns.index[window:]
    
    entropies = []
    mean_corrs = []
    
    print(f"Computing metrics (Window={window})...")
    
    for i in range(window, len(returns)):
        window_data = returns.iloc[i-window:i]
        corr_matrix = window_data.corr().values
        
        # 1. Mean Correlation (Off-diagonal)
        mask = np.ones_like(corr_matrix, dtype=bool)
        np.fill_diagonal(mask, False)
        mean_rho = np.mean(corr_matrix[mask])
        mean_corrs.append(mean_rho)
        
        # 2. Geometric Entropy
        eigvals = np.linalg.eigvalsh(corr_matrix)
        eigvals = eigvals[eigvals > 1e-10]
        eigvals = eigvals / np.sum(eigvals)
        s = -np.sum(eigvals * np.log(eigvals + 1e-20))
        entropies.append(s)
        
    S = pd.Series(entropies, index=dates, name='Entropy')
    Rho = pd.Series(mean_corrs, index=dates, name='MeanCorr')
    
    # Re-align VIX to the windowed dates
    VIX = vix.loc[dates]
    if isinstance(VIX, pd.DataFrame): VIX = VIX.iloc[:, 0]
    
    # -------------------------------------------------------------------------
    # 1. Collinearity Check
    # -------------------------------------------------------------------------
    corr_s_rho, _ = pearsonr(S, Rho)
    print(f"\n>>> Correlation(Entropy, MeanCorr) = {corr_s_rho:.4f}")
    
    if abs(corr_s_rho) > 0.9:
        print("WARNING: High collinearity! Entropy is mostly just correlation.")
    else:
        print("GOOD: Entropy contains distinct topological information.")
        
    # -------------------------------------------------------------------------
    # 2. Lead-Lag Analysis (Granger-style)
    # -------------------------------------------------------------------------
    # Check max correlation at different lags with VIX
    lags = range(-20, 21) # Days
    
    xcorr_s = [S.corr(VIX.shift(-l)) for l in lags]
    xcorr_rho = [Rho.corr(VIX.shift(-l)) for l in lags]
    
    # Find peak lag (When does metric best predict VIX?)
    # Note: S is usually negative correlated with VIX (Low Entropy -> High Fear)
    # So we look for min correlation (most negative)
    
    peak_lag_s = lags[np.argmin(xcorr_s)]
    peak_val_s = np.min(xcorr_s)
    
    # Rho is positive correlated with VIX
    peak_lag_rho = lags[np.argmax(xcorr_rho)]
    peak_val_rho = np.max(xcorr_rho)
    
    print(f"\n--- Lead-Lag Analysis (Predicting VIX) ---")
    print(f"Entropy: Peak Corr = {peak_val_s:.3f} at Lag = {peak_lag_s} days")
    print(f"MeanCorr: Peak Corr = {peak_val_rho:.3f} at Lag = {peak_lag_rho} days")
    
    # Interpretation: Lag < 0 means Metric LEADS VIX.
    
    # -------------------------------------------------------------------------
    # Plotting
    # -------------------------------------------------------------------------
    os.makedirs('figures', exist_ok=True)
    fig, ax1 = plt.subplots(figsize=(10, 6))
    
    color = 'tab:blue'
    ax1.set_xlabel('Lag (Days)')
    ax1.set_ylabel('Correlation with VIX', color=color)
    ax1.plot(lags, xcorr_s, color=color, label='Entropy vs VIX (Negative is good)')
    ax1.plot(lags, xcorr_rho, color='tab:orange', label='MeanCorr vs VIX')
    
    plt.axvline(x=0, color='black', linestyle='--')
    plt.title("Lead-Lag Analysis: Does Entropy Predict Crisis Faster?")
    plt.legend()
    plt.grid(True)
    
    plt.savefig('figures/lead_lag_analysis.png', dpi=300)
    print("Saved analysis to figures/lead_lag_analysis.png")

if __name__ == "__main__":
    analyze_incremental_value()
