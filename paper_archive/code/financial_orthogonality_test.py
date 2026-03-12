import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import yfinance as yf
import os
from sklearn.feature_selection import mutual_info_regression
from sklearn.linear_model import LinearRegression
from scipy.stats import pearsonr

"""
EXP 71: Financial Orthogonality Test
------------------------------------
Critique addressed: "Is Causal Horizon just Volatility in disguise?"
Hypothesis: Tau_eff contains structural information orthogonal to simple volatility.

Protocol:
1. Compute Tau_eff (Horizon) and Realized Volatility (Vol).
2. Regress Tau_eff against Vol: Tau = alpha + beta * Vol + epsilon.
3. Analyze Residuals (epsilon):
   - Do they still contain predictive power for future crises?
   - If yes, Tau_eff is NOT just Volatility.
"""

def fetch_real_data(ticker="SPY", start="2006-01-01", end="2012-01-01"):
    print(f"Fetching real data for {ticker} from {start} to {end}...")
    try:
        data = yf.download(ticker, start=start, end=end, progress=False)
        if len(data) == 0:
            raise ValueError("No data downloaded")
        if isinstance(data, pd.DataFrame):
            if 'Close' in data.columns:
                return data['Close'].squeeze()
            else:
                return data.iloc[:, 0]
        return data
    except Exception as e:
        print(f"Failed to download data: {e}")
        dates = pd.date_range(start=start, end=end, freq='B')
        prices = 100 * np.exp(np.cumsum(np.random.normal(0, 0.01, len(dates))))
        return pd.Series(prices, index=dates)

def compute_metrics(series, window_size=120):
    """Computes Horizon, Realized Volatility, and Returns"""
    horizons = []
    dates = []
    
    returns = np.log(series / series.shift(1)).dropna()
    abs_returns = np.abs(returns)
    
    # 1. Realized Volatility (Rolling Std)
    # We want it aligned with the END of the window, same as Horizon
    rolling_vol = returns.rolling(window=window_size).std()
    
    print(f"Computing Causal Horizon (Window={window_size})...")
    step = 5
    
    # Storage for aligned data
    aligned_vol = []
    aligned_dates = []
    
    for t in range(window_size, len(returns), step):
        segment = abs_returns.iloc[t-window_size:t]
        current_date = returns.index[t]
        
        # --- Compute Tau_eff ---
        mi_scores = []
        max_lag = 30
        
        for lag in range(1, max_lag + 1):
            series_now = segment.iloc[lag:]
            series_prev = segment.shift(lag).iloc[lag:]
            
            X = series_prev.values.reshape(-1, 1)
            y = series_now.values
            
            mi = mutual_info_regression(X, y, discrete_features=False, random_state=42)[0]
            mi_scores.append(mi)
            
        mi_scores = np.array(mi_scores)
        total_info = np.sum(mi_scores) + 1e-9
        lags = np.arange(1, max_lag + 1)
        tau_eff = np.sum(lags * mi_scores) / total_info
        
        horizons.append(tau_eff)
        aligned_dates.append(current_date)
        aligned_vol.append(rolling_vol.iloc[t])
        
    return pd.DataFrame({
        'Date': aligned_dates,
        'Horizon': horizons,
        'Volatility': aligned_vol
    }).set_index('Date')

def run_orthogonality_test():
    # 1. Load Data
    prices = fetch_real_data("SPY", start="2006-01-01", end="2012-01-01")
    vix = fetch_real_data("^VIX", start="2006-01-01", end="2012-01-01")
    
    # 2. Compute Metrics
    df = compute_metrics(prices)
    
    # 3. Align VIX
    # VIX is already annualized vol, need to align dates
    df['VIX'] = vix.reindex(df.index, method='nearest')
    
    # Drop NaNs
    df = df.dropna()
    
    # --- ANALYSIS 1: Correlation Matrix ---
    print("\n--- Correlation Matrix ---")
    corr = df.corr()
    print(corr)
    
    # --- ANALYSIS 2: Orthogonalization (Residuals) ---
    # Model: Horizon = alpha + beta * VIX + epsilon
    # We want to isolate 'epsilon' (Structure) from 'VIX' (Panic/Vol)
    
    X = df['VIX'].values.reshape(-1, 1)
    y = df['Horizon'].values
    
    reg = LinearRegression()
    reg.fit(X, y)
    
    predicted_horizon_from_vix = reg.predict(X)
    residuals = y - predicted_horizon_from_vix # This is the "Pure Structure" component
    
    df['Residuals'] = residuals
    
    print(f"\nRegression: Horizon = {reg.intercept_:.2f} + {reg.coef_[0]:.4f} * VIX")
    print(f"R-squared: {reg.score(X, y):.4f}")
    
    # --- ANALYSIS 3: Predictive Power of Residuals ---
    # Can the "Residuals" (Non-VIX part of Horizon) still predict future volatility?
    
    # Target: Future VIX (e.g., 20 days later)
    future_vix = df['VIX'].shift(-4) # 4 steps * 5 days = 20 days
    
    valid_mask = ~np.isnan(future_vix)
    
    corr_resid_future = pearsonr(df['Residuals'][valid_mask], future_vix[valid_mask])
    print(f"\nCorrelation(Residuals_t, VIX_{{t+20d}}): {corr_resid_future[0]:.4f} (p={corr_resid_future[1]:.2e})")
    
    # --- PLOTTING ---
    os.makedirs('figures', exist_ok=True)
    fig = plt.figure(figsize=(14, 10))
    gs = fig.add_gridspec(2, 2)
    
    # Plot A: Scatter VIX vs Horizon
    ax1 = fig.add_subplot(gs[0, 0])
    sns.regplot(x='VIX', y='Horizon', data=df, ax=ax1, scatter_kws={'alpha':0.3}, line_kws={'color':'red'})
    ax1.set_title(f'A. Horizon vs VIX (R² = {reg.score(X, y):.2f})')
    ax1.set_xlabel('Market Fear (VIX)')
    ax1.set_ylabel('Causal Horizon (Tau)')
    
    # Plot B: Time Series of Residuals during Crisis
    ax2 = fig.add_subplot(gs[0, 1])
    # Zoom in on 2008
    subset = df['2008-01-01':'2009-01-01']
    
    ax2.plot(subset.index, subset['Horizon'], label='Raw Horizon', color='blue', alpha=0.5)
    ax2.plot(subset.index, subset['Residuals'], label='Orthogonal Residuals (Non-VIX)', color='green', linewidth=2)
    ax2.axhline(0, color='black', linestyle='--')
    ax2.set_title('B. De-trended Horizon during 2008 Crisis')
    ax2.legend()
    
    # Plot C: Lead-Lag Analysis of RESIDUALS
    ax3 = fig.add_subplot(gs[1, :])
    
    lags = np.arange(0, 12) # 0 to 60 days (steps of 5)
    corrs_raw = []
    corrs_resid = []
    
    target = df['VIX'].values
    
    for k in lags:
        # Shift target backwards (Correlation between Signal_t and Target_{t+k})
        if k == 0:
            t_k = target
            h_raw = df['Horizon'].values
            h_res = df['Residuals'].values
        else:
            t_k = target[k:]
            h_raw = df['Horizon'].values[:-k]
            h_res = df['Residuals'].values[:-k]
            
        corrs_raw.append(np.abs(np.corrcoef(h_raw, t_k)[0,1]))
        corrs_resid.append(np.abs(np.corrcoef(h_res, t_k)[0,1]))
        
    days = lags * 5
    ax3.plot(days, corrs_raw, 'b-o', label='Raw Horizon')
    ax3.plot(days, corrs_resid, 'g-s', label='Orthogonal Residuals (After removing VIX)')
    ax3.set_xlabel('Prediction Horizon (Days Ahead)')
    ax3.set_ylabel('Correlation with Future VIX')
    ax3.set_title('C. Predictive Power: Does the "Non-VIX" part still predict the crisis?')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('figures/orthogonality_proof.png', dpi=300)
    print("Saved orthogonality proof to figures/orthogonality_proof.png")

if __name__ == "__main__":
    run_orthogonality_test()

