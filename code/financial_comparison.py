import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf
import pandas as pd
from scipy.stats import linregress
import os

"""
EXP 110: Causal Horizon vs Hurst Exponent
-----------------------------------------
Goal: Demonstrate that effective causal horizon (tau_eff) is a more sensitive 
and physically meaningful metric for crisis detection than the classical Hurst exponent.

Data: S&P 500 (GSPC) 2000-2023 (Covering Dot-com, 2008, Covid)
"""

def calculate_hurst(ts, max_lag=20):
    """
    Compute Hurst exponent via R/S analysis or simpler Variance method.
    Var(tau) ~ tau^(2H)
    """
    lags = range(2, max_lag)
    tau = [np.sqrt(np.std(ts[lag:] - ts[:-lag])) for lag in lags]
    # polyfit log(lags) vs log(tau)
    # log(sigma) = H * log(lag) + c
    try:
        m, c, _, _, _ = linregress(np.log(lags), np.log(tau))
        return m
    except:
        return 0.5

def calculate_tau_eff(ts, max_lag=50, noise_threshold=0.1):
    """
    Compute Effective Causal Horizon using Spectral Center of Mass (Weighted Average).
    Formula from Paper (Eq 268): tau_eff = Sum(tau * I(tau)) / Sum(I(tau))
    
    We use Squared Autocorrelation (rho^2) as a proxy for Mutual Information I,
    based on the Gaussian approximation I ~ -0.5 * log(1 - rho^2) ~ rho^2/2.
    This ensures strict consistency with the "Spectral Center of Mass" definition.
    """
    # Fix 1: Ensure float input and remove infinite values
    ts = np.asarray(ts, dtype=float)
    if np.any(np.isinf(ts)) or np.any(np.isnan(ts)):
        return 0
        
    n = len(ts)
    if n < 5: return 0
    
    lags = np.arange(1, max_lag + 1)
    correlations = []
    
    for lag in lags:
        if lag >= n: 
            correlations.append(0)
            continue
            
        # Compute Correlation
        # Note: ts is usually Volatility (abs returns), so correlation is positive
        try:
            # Fix 2: Manual correlation to avoid NaN from np.corrcoef when var is 0
            x = ts[:-lag]
            y = ts[lag:]
            if np.std(x) < 1e-9 or np.std(y) < 1e-9:
                 c = 0
            else:
                 c = np.corrcoef(x, y)[0, 1]
                 
            if np.isnan(c): c = 0
        except:
            c = 0
        correlations.append(c)
    
    correlations = np.array(correlations)
    
    # Weight by Information Content (rho^2)
    weights = correlations ** 2
    
    total_info = np.sum(weights)
    
    if total_info < 1e-9:
        return 0
        
    # Weighted Average Lag (Center of Mass)
    tau_eff = np.sum(lags * weights) / total_info
    
    return tau_eff

def run_comparison():
    print("Downloading S&P 500 Data...")
    try:
        sp500 = yf.download("^GSPC", start="2006-01-01", end="2010-01-01", progress=False)
        # Fix 3: Handle multi-level column index if present (common in new yfinance)
        if isinstance(sp500.columns, pd.MultiIndex):
            prices = sp500[('Close', '^GSPC')].values
        else:
            prices = sp500['Close'].values
            
        dates = sp500.index
    except Exception as e:
        print(f"Download Error: {e}")
        # Fallback to synthetic

        print("Download failed. Using synthetic Geometric Brownian Motion.")
        T = 1000
        t = np.linspace(0, 1, T)
        prices = np.exp(np.cumsum(np.random.randn(T)*0.02))
        dates = range(T)

    # Returns
    returns = np.diff(np.log(prices))
    
    # Check data validity
    print(f"Data shape: {prices.shape}")
    print(f"Sample prices: {prices[:5]}")
    if len(prices) < 200:
        print("Error: Not enough data downloaded.")
        return

    # Sliding Window Analysis
    window = 120 # Increased window size for better correlation estimates
    step = 5
    
    hurst_history = []
    tau_history = []
    date_history = []
    price_history = []
    
    print("Calculating metrics...")
    for i in range(0, len(returns) - window, step):
        segment = returns[i:i+window]
        
        # Hurst (on prices or returns? Hurst usually on returns/volatility or prices directly)
        # Standard Hurst is for random walk check.
        # Let's use Prices segment for Hurst (Trend persistence)
        price_segment = prices[i:i+window]
        h = calculate_hurst(price_segment)
        
        # Tau Eff (on Returns - Information memory)
        # We want to measure how long return correlations last (Vol clustering)
        # Or Price memory?
        # Let's use Price memory (Autocorrelation of levels) - Dangerous (non-stationary)
        # Better: Volatility memory (Abs returns)
        vol_segment = np.abs(segment)
        
        # DEBUG: Check if vol_segment has variance
        if np.std(vol_segment) < 1e-9:
             tau = 0
        else:
             tau = calculate_tau_eff(vol_segment, max_lag=30)
        
        hurst_history.append(h)
        tau_history.append(tau)
        date_history.append(dates[i+window])
        price_history.append(prices[i+window])

        
    # Check stats of generated metrics
    tau_arr = np.array(tau_history)
    hurst_arr = np.array(hurst_history)
    print(f"\nMetric Statistics:")
    print(f"Tau: Min={tau_arr.min():.4f}, Max={tau_arr.max():.4f}, Mean={tau_arr.mean():.4f}")
    print(f"Hurst: Min={hurst_arr.min():.4f}, Max={hurst_arr.max():.4f}, Mean={hurst_arr.mean():.4f}")
    
    # Smoothing
    def smooth(x, w=5):
        return np.convolve(x, np.ones(w)/w, mode='valid')
        
    h_smooth = smooth(hurst_history)
    t_smooth = smooth(tau_history)
    d_smooth = date_history[len(date_history)-len(h_smooth):]
    p_smooth = price_history[len(price_history)-len(h_smooth):]
    
    # --- Plotting ---
    os.makedirs('figures', exist_ok=True)
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10), gridspec_kw={'height_ratios': [1, 2]})
    
    # 1. Timeline
    ax1.plot(d_smooth, p_smooth, 'k-', label='S&P 500')
    ax1.set_ylabel('Index Level')
    ax1.set_title('Financial Crisis Timeline (2008)', fontsize=12)
    ax1.grid(True, alpha=0.3)
    
    # Color background for crisis (Lehman Sep 2008)
    # 2008-09-15
    import datetime
    crisis_start = pd.Timestamp("2008-09-01")
    crisis_end = pd.Timestamp("2008-12-01")
    ax1.axvspan(crisis_start, crisis_end, color='red', alpha=0.2, label='Lehman Crash')
    ax1.legend()
    
    # 2. Phase Portrait (Hurst vs Tau)
    # We want to show the TRAJECTORY
    sc = ax2.scatter(h_smooth, t_smooth, c=range(len(h_smooth)), cmap='coolwarm', s=50, alpha=0.8)
    
    # Add arrows to show time direction (Trajectory)
    # Only adding arrows every N steps to avoid clutter
    step_arrow = 3
    for i in range(0, len(h_smooth)-1, step_arrow):
       # Calculate delta
       dx = h_smooth[i+1] - h_smooth[i]
       dy = t_smooth[i+1] - t_smooth[i]
       
       # Normalize length for better visualization if needed, but simple quiver/arrow is okay
       # Use annotate arrow props for better control than ax.arrow
       if np.sqrt(dx**2 + dy**2) > 0.001: # Only draw if moved significantly
           ax2.annotate('', xy=(h_smooth[i+1], t_smooth[i+1]), xytext=(h_smooth[i], t_smooth[i]),
                        arrowprops=dict(arrowstyle="->", color='gray', alpha=0.3, lw=1))
    
    
    # Highlight Crisis Points
    # Find points inside the crisis window
    crisis_indices = [i for i, d in enumerate(d_smooth) if crisis_start <= d <= crisis_end]
    if crisis_indices:
        ax2.scatter([h_smooth[i] for i in crisis_indices], 
                   [t_smooth[i] for i in crisis_indices], 
                   color='red', s=100, marker='x', label='Crisis Phase')
    
    ax2.set_xlabel('Hurst Exponent (Memory)', fontsize=12)
    ax2.set_ylabel('Effective Causal Horizon $\\tau_{eff}$ (Information)', fontsize=12)
    ax2.set_title('Phase Space: Horizon Collapse vs Hurst', fontsize=14)
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    
    # Annotation
    ax2.text(0.1, 0.9, 'Stable Market:\nHigh H, High $\\tau$', transform=ax2.transAxes, color='blue')
    ax2.text(0.1, 0.1, 'Crash (Event Horizon):\n$\\tau \\to 0$, H varies', transform=ax2.transAxes, color='red')
    
    plt.tight_layout()
    plt.savefig('figures/financial_comparison.png', dpi=300)
    print("Saved comparison to figures/financial_comparison.png")

if __name__ == "__main__":
    run_comparison()
