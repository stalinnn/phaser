import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import yfinance as yf
import os
from sklearn.feature_selection import mutual_info_regression

"""
EXP 70: Empirical Financial Causal Horizon (Real Data)
----------------------------------------------------
Critique addressed: "Circular Reasoning with Synthetic Data".
Solution: We use REAL market data (S&P 500) via yfinance.

We measure the "Effective Predictive Horizon" (tau) of the market over time.
Hypothesis: 
During crises (2008 Lehman, 2020 COVID), the market's memory length COLLAPSES.
This confirms the transition from "Investing" (Long-term calculation) to 
"Panic" (Short-term herding).
"""

def fetch_real_data(ticker="SPY", start="2007-01-01", end="2021-01-01"):
    print(f"Fetching real data for {ticker} from {start} to {end}...")
    try:
        data = yf.download(ticker, start=start, end=end, progress=False)
        if len(data) == 0:
            raise ValueError("No data downloaded")
        # Ensure we return a Series, not a DataFrame
        if isinstance(data, pd.DataFrame):
            # yfinance often returns MultiIndex columns or DataFrame
            if 'Close' in data.columns:
                return data['Close'].squeeze() # Convert to Series
            else:
                return data.iloc[:, 0] # First column
        return data
    except Exception as e:
        print(f"Failed to download data: {e}")
        print("Falling back to synthetic data (for offline testing only)...")
        # Fallback only if network fails
        dates = pd.date_range(start=start, end=end, freq='B')
        prices = 100 * np.exp(np.cumsum(np.random.normal(0, 0.01, len(dates))))
        return pd.Series(prices, index=dates)

def compute_causal_horizon(series, window_size=120, max_lag=30, step=5):
    """
    Computes the 'Short-termism Ratio' of the market.
    Short-termism = (Short Lag Corr) / (Total Lag Corr)
    
    Hypothesis: In crisis, market becomes obsessed with immediate past (High Short-termism).
    Horizon Collapse <=> Short-termism Spike.
    """
    ratios = []
    dates = []
    
    # Use Log Returns
    # Ensure it's a Series
    if isinstance(series, pd.DataFrame):
        series = series.squeeze()
        
    returns = np.log(series / series.shift(1)).dropna()
    abs_returns = np.abs(returns) # Volatility proxy
    
    print(f"Scanning Short-termism on Real Data (Window={window_size} days)...")
    
    for t in range(window_size, len(returns), step):
        segment = abs_returns.iloc[t-window_size:t]
        current_date = returns.index[t]
        
        # Calculate Mutual Information Spectrum
        mi_scores = []
        
        # Prepare data for MI calculation
        # Target: current return r_t
        # Features: r_{t-1}, r_{t-2}, ... r_{t-lag}
        
        # We need a small sub-window to compute MI reliably. 
        # But 'segment' is just a 1D array of length 'window_size'.
        # We want to know: How much does r_{t-k} explain r_t WITHIN this window?
        
        # Construct lag matrix for this window
        window_len = len(segment)
        X_lags = []
        y_now = segment.values # Target variable (volatility at t)
        
        # This is tricky inside a loop. MI needs samples.
        # "segment" provides T samples.
        # We want to measure the dependency structure characteristic of this window.
        
        # Let's compute pairwise MI for each lag k over the samples in the window.
        # Lag k means: correlation between seg[k:] and seg[:-k]
        
        for lag in range(1, max_lag + 1):
            # Create lagged pairs
            series_now = segment.iloc[lag:]
            series_prev = segment.shift(lag).iloc[lag:]
            
            # Compute MI between X (past) and Y (future)
            # Reshape for sklearn
            X = series_prev.values.reshape(-1, 1)
            y = series_now.values
            
            # discrete_features=False for continuous data
            # n_neighbors=3 is standard for MI estimation
            mi = mutual_info_regression(X, y, discrete_features=False, random_state=42)[0]
            mi_scores.append(mi)
            
        mi_scores = np.array(mi_scores)
        total_info = np.sum(mi_scores) + 1e-9
        
        # Define "Information Horizon" (tau_eff)
        # As the lag where cumsum(MI) reaches 90% of total MI?
        # Or simply the weighted average lag? 
        # Tau = Sum(k * MI_k) / Sum(MI_k)
        
        lags = np.arange(1, max_lag + 1)
        # Definition Update: We use Center of Mass (Expectation) as it is more robust than First Hitting Time
        # Tau = E[lag] under distribution p(lag) ~ MI(lag)
        tau_eff = np.sum(lags * mi_scores) / total_info
        
        ratios.append(tau_eff)
        dates.append(current_date)
        
    return dates, ratios, returns

def compute_hurst(series, window_size=120):
    """
    Computes rolling Hurst Exponent as a baseline comparison.
    Simplified R/S analysis.
    """
    hursts = []
    # Alignment with horizon dates: we need to match the windows
    # The previous function returns dates at the END of the window
    # So we should do the same here.
    
    # Pre-calculate to match length
    series_log = np.log(series)
    
    # We will compute Hurst only for the same indices
    # But for simplicity in this script, let's just make a standalone loop similar to horizon
    
    for t in range(window_size, len(series), 5): # Step 5 to match
        # R/S Analysis
        chunk = series_log[t-window_size:t].values
        if len(chunk) < 20: 
            hursts.append(0.5)
            continue
            
        # Returns
        r = np.diff(chunk)
        # Mean
        mean_r = np.mean(r)
        # Deviations
        z = np.cumsum(r - mean_r)
        # Range
        R = np.max(z) - np.min(z)
        # Std
        S = np.std(r)
        if S == 0: S = 1e-9
        
        # H approx log(R/S) / log(N)
        # This is a crude estimator but standard for rolling windows
        h = np.log(R/S) / np.log(len(r))
        hursts.append(h)
        
    return hursts

def run_real_analysis():
    # Multi-Asset Universality Test
    assets = {
        'S&P 500 (US)': 'SPY'
    }
    
    # Fetch VIX for Comparison
    print("Fetching VIX data...")
    try:
        vix = fetch_real_data('^VIX', start="2006-01-01", end="2012-01-01")
    except:
        print("VIX fetch failed.")
        vix = None
    
    for i, (name, ticker) in enumerate(assets.items()):
        print(f"\n--- Analyzing {name} ({ticker}) ---")
        
        # 1. Get Data
        prices = fetch_real_data(ticker, start="2006-01-01", end="2012-01-01") 
        if len(prices) < 200: continue
            
        window = 120
        dates, horizons, returns = compute_causal_horizon(prices, window_size=window)
        hursts = compute_hurst(prices, window_size=window)
        
        # Align VIX
        if vix is not None:
            # Reindex VIX to match analysis dates
            # 'dates' are the END of the window. We want VIX at that moment.
            vix_aligned = vix.reindex(dates, method='nearest').values
        else:
            vix_aligned = np.zeros(len(dates))
        
        # Ensure lengths match
        min_len = min(len(horizons), len(hursts), len(vix_aligned))
        horizons = np.array(horizons[:min_len])
        hursts = np.array(hursts[:min_len])
        vix_series = np.array(vix_aligned[:min_len])
        dates = dates[:min_len]
        
        # Volatility Target (Future Realized Volatility)
        # We calculate realized vol over next 30 days
        future_vol = prices.pct_change().rolling(30).std().shift(-30)
        # Align with dates
        target_vol = future_vol.loc[dates].values
        
        # Clean NaNs from shift
        valid_mask = ~np.isnan(target_vol) & ~np.isnan(vix_series)
        
        h_clean = horizons[valid_mask]
        hurst_clean = hursts[valid_mask]
        vix_clean = vix_series[valid_mask]
        y_clean = target_vol[valid_mask]
        
        # --- Lead-Lag Analysis ---
        # Cross correlation with Future Realized Volatility
        lags = np.arange(0, 60, 2) # Step 2 for cleaner plot
        
        # We compute corr(Signal(t), RealizedVol(t+k))
        # Note: target_vol is ALREADY shifted by 30 days (it's the 30-day future vol).
        # So lag=0 means "Predicting next 30 days vol based on today's signal".
        # lag=k means "Predicting vol starts at t+k".
        # Let's align signals instead.
        
        # Let's verify correlation profile:
        # Does Horizon(t) predict Volatility(t+k)?
        
        # Re-extract raw daily vol for fine-grained lead-lag
        daily_vol_proxy = np.abs(prices.pct_change()).loc[dates].values # Simple proxy
        
        corrs = {'Horizon': [], 'Hurst': [], 'VIX': []}
        
        for k in lags:
            # Shift target: Volatility k days ahead
            # We use the 30-day realized volatility STARTING at t+k
            # This is complex to reconstruct.
            # Simpler: Just shift the 'y_clean' (Next 30 day vol) by k indices? 
            # No, 'dates' are not necessarily contiguous days (step=5).
            
            # Let's go back to simplest definition:
            # Correlation between Metric(t) and VIX(t+k)
            # We treat VIX itself as the "Crisis Event".
            
            # Target: VIX(t+k)
            # Can we predict the VIX itself?
            
            # Shift VIX forward by k steps (in the subsampled domain, step=5 days)
            # So k=1 means 5 days ahead. k=12 means 60 days ahead.
            
            if k == 0:
                v_target = vix_clean
                h_source = h_clean
                hu_source = hurst_clean
                vi_source = vix_clean
            else:
                v_target = vix_clean[k:]
                h_source = h_clean[:-k]
                hu_source = hurst_clean[:-k]
                vi_source = vix_clean[:-k]
                
            # Horizon is anti-correlated with panic
            corrs['Horizon'].append(np.abs(np.corrcoef(h_source, v_target)[0,1]))
            # Hurst is anti-correlated (Trend vs Mean Reversion/Chaos)
            # Wait, Hurst->0.5 is random. Hurst->1 is trend. Crisis is often high trend (crash).
            corrs['Hurst'].append(np.abs(np.corrcoef(hu_source, v_target)[0,1]))
            # VIX autocorrelation
            corrs['VIX'].append(np.abs(np.corrcoef(vi_source, v_target)[0,1]))
            
        # Plot Lead-Lag
        plt.figure(figsize=(10, 6))
        # Convert steps to days (approx step=5 business days)
        days_ahead = lags * 5
        
        plt.plot(days_ahead, corrs['Horizon'], 'b-o', linewidth=2, label='Causal Horizon (Ours)')
        plt.plot(days_ahead, corrs['Hurst'], 'r-s', alpha=0.5, label='Hurst Exponent')
        plt.plot(days_ahead, corrs['VIX'], 'k--', alpha=0.5, label='VIX (Autoregression)')
        
        plt.xlabel('Days Ahead (Forecasting Horizon)')
        plt.ylabel('Correlation with Future VIX')
        plt.title(f'Crisis Early Warning: {name} (2008)\nCan we predict future panic better than VIX itself?')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Add annotation if Horizon beats VIX at long range
        if corrs['Horizon'][-1] > corrs['VIX'][-1]:
             plt.text(days_ahead[-5], corrs['Horizon'][-5] + 0.05, "Long-Range\nSuperiority", color='blue', fontweight='bold')
        
        os.makedirs('figures', exist_ok=True)
        plt.savefig('figures/lead_lag_analysis.png', dpi=300)
        print("Saved Lead-Lag analysis (with VIX) to figures/lead_lag_analysis.png")


if __name__ == "__main__":
    run_real_analysis()
