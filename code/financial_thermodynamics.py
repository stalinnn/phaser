import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import NMF
from sklearn.linear_model import LinearRegression
from scipy.stats import pearsonr
import os

"""
FINANCIAL THERMODYNAMICS & ADIABATIC ATTENTION
----------------------------------------------
Tier 1 Upgrade:
1. Data Scale: Expanded to S&P 100+ (approximated list for speed) & extended timeline (2000-2024).
2. Mechanism Check: Added Amihud Illiquidity proxy to correlate with Order Parameter.
3. Benchmark: Added simple Rolling Correlation mean as a benchmark.
"""

def download_data():
    # Expanded Ticker List (Top ~100 historically significant US stocks)
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
    # Remove duplicates
    tickers = list(set(tickers))
    
    start_date = "2000-01-01"
    end_date = "2024-01-01"
    
    print(f"Downloading data for {len(tickers)} tickers from {start_date} to {end_date}...")
    try:
        # We need both Close (for returns) and Volume (for Amihud Illiquidity)
        # Using group_by='ticker' in yfinance is better for multi-index, 
        # but auto_adjust=True helps.
        # Let's download normally and handle the multi-index.
        data_raw = yf.download(tickers, start=start_date, end=end_date, progress=True, auto_adjust=True)
        
        # Check structure
        if 'Close' in data_raw.columns:
            close_data = data_raw['Close']
            volume_data = data_raw['Volume']
        else:
            # Maybe flat structure if 1 ticker? No, we have list.
            # yfinance recent versions might return just one level if columns are not aligned
            # But usually it's MultiIndex (Price, Ticker)
            # Let's try to assume it worked.
            close_data = data_raw # If only one type requested, but we need volume too.
            # Actually yf.download returns a DataFrame with MultiIndex columns (PriceType, Ticker)
            # We need to be careful.
            pass

        # Fill NaNs
        close_data = close_data.ffill().dropna(axis=1, how='all') # Drop tickers with ALL NaNs
        volume_data = volume_data.ffill().dropna(axis=1, how='all')
        
        # Keep only intersection of columns
        common_cols = close_data.columns.intersection(volume_data.columns)
        close_data = close_data[common_cols]
        volume_data = volume_data[common_cols]
        
        # FIX: Instead of dropping all rows with ANY NaN (which truncates history to the newest stock),
        # we drop columns (stocks) that don't have enough history.
        # We want data from 2000/2002. Let's require at least 80% of the timeline.
        threshold = int(len(close_data) * 0.8)
        close_data = close_data.dropna(axis=1, thresh=threshold)
        volume_data = volume_data[close_data.columns] # Sync columns
        
        # Now we can safely drop rows that still have NaNs (mostly weekends or holidays matching)
        # or fill remaining small gaps
        close_data = close_data.ffill().bfill().dropna()
        volume_data = volume_data.ffill().bfill().dropna()
        
        # Align indices
        common_idx = close_data.index.intersection(volume_data.index)
        close_data = close_data.loc[common_idx]
        volume_data = volume_data.loc[common_idx]
        
        print(f"  Data shape after cleaning: {close_data.shape} (Dates, Tickers)")
        return close_data, volume_data
        
    except Exception as e:
        print(f"Data download failed: {e}")
        return None, None

class AdiabaticAttention:
    def __init__(self, n_components=3, slow_decay=0.98):
        self.n_components = n_components
        self.slow_decay = slow_decay 
        self.A_slow = None 
        self.K_current = None
        
    def fit_step(self, returns_window):
        # 1. Compute Instantaneous Interaction (A_fast)
        if returns_window.shape[1] < 5: return None, None # Safety
        
        # Shrinkage correlation could be better for large N, but simple corr for now
        A_fast = returns_window.corr().values
        A_fast = np.nan_to_num(A_fast)
        
        # 2. Update Slow Manifold (A_slow)
        if self.A_slow is None:
            self.A_slow = A_fast
        else:
            self.A_slow = self.slow_decay * self.A_slow + (1 - self.slow_decay) * A_fast
            
        # 3. Extract Keys (K)
        A_slow_shifted = self.A_slow + 1.0 
        
        try:
            model = NMF(n_components=self.n_components, init='random', random_state=42, max_iter=200, tol=1e-3)
            W = model.fit_transform(A_slow_shifted)
            H = model.components_
            self.K_current = H.T
        except:
            # Fallback if NMF fails
            self.K_current = np.random.rand(A_fast.shape[0], self.n_components)

        # 4. Solve for Instantaneous Query (Q)
        A_fast_shifted = A_fast + 1.0
        lambda_reg = 0.5 # Higher regularization for larger matrices
        K = self.K_current
        
        try:
            K_inv = np.linalg.inv(K.T @ K + lambda_reg * np.eye(self.n_components))
            Q_t = A_fast_shifted @ K @ K_inv
        except:
            Q_t = np.random.rand(A_fast.shape[0], self.n_components)
            
        return Q_t, self.K_current

def calculate_parisi_order(Q):
    if Q is None: return 0, 0
    norm = np.linalg.norm(Q, axis=1, keepdims=True)
    Q_norm = Q / (norm + 1e-9)
    overlaps = Q_norm @ Q_norm.T
    n = overlaps.shape[0]
    off_diag = overlaps[np.triu_indices(n, k=1)]
    m = np.mean(off_diag)
    chi_SG = np.var(off_diag)
    return m, chi_SG

def calculate_amihud_illiquidity(close_window, volume_window):
    # Amihud = |Return| / (Price * Volume)
    # We approximate "Dollar Volume" as Close * Volume
    # Returns 1-day returns in the window
    returns = close_window.pct_change().abs()
    dollar_vol = close_window * volume_window
    
    # Avoid div by zero
    dollar_vol = dollar_vol.replace(0, np.nan)
    
    illiq = returns / dollar_vol
    # Average across all assets in the window, then average across time in the window
    # Scale up for readability (x 1e6)
    return illiq.mean().mean() * 1e9 

def run_simulation_logic(returns, close_prices=None, volumes=None, label="Real Data"):
    window_size = 60 # 3 months
    step = 20 # Monthly steps for speed
    solver = AdiabaticAttention(n_components=5, slow_decay=0.95) # More components for more assets
    
    results = []
    
    print(f"  Running loop for {label} ({len(returns)} steps)...")
    
    for t in range(window_size, len(returns), step):
        current_date = returns.index[t]
        window_returns = returns.iloc[t-window_size:t]
        
        # 1. Attention Model
        Q_t, K_t = solver.fit_step(window_returns)
        order_m, susceptibility = calculate_parisi_order(Q_t)
        
        # 2. Benchmark 1: Rolling Correlation (Standard)
        # Average pairwise correlation in the window
        avg_corr = window_returns.corr().values[np.triu_indices(window_returns.shape[1], k=1)].mean()
        
        # 3. Mechanism Check: Amihud Illiquidity (if data available)
        amihud = np.nan
        if close_prices is not None and volumes is not None:
             window_close = close_prices.iloc[t-window_size:t]
             window_vol = volumes.iloc[t-window_size:t]
             amihud = calculate_amihud_illiquidity(window_close, window_vol)
        
        results.append({
            'Date': current_date,
            'Order_Parameter_M': order_m,
            'Susceptibility_Chi': susceptibility,
            'Benchmark_Avg_Corr': avg_corr,
            'Amihud_Illiquidity': amihud
        })
        
        if t % 500 == 0:
            print(f"    Progress: {t}/{len(returns)}")
    
    return pd.DataFrame(results).set_index('Date')

def run_thermodynamics():
    close_data, volume_data = download_data()
    if close_data is None:
        return

    # Calculate Log Returns
    returns = np.log(close_data / close_data.shift(1)).dropna()
    # Align volume data
    volume_data = volume_data.loc[returns.index]
    close_data = close_data.loc[returns.index]
    
    # 1. Run Main Simulation
    print("Running Adiabatic Evolution (Real Data)...")
    df_res = run_simulation_logic(returns, close_data, volume_data, label="Real Data")
    
    # 2. Mechanism Validation: Correlation with Liquidity
    # We expect Order Parameter (Herding) to be High when Liquidity is Low (Amihud is High)
    # So Correlation(M, Amihud) should be POSITIVE? 
    # Wait, High Amihud = Low Liquidity.
    # High Order M = Crisis/Herding.
    # So Yes, Positive Correlation.
    # But wait, T = 1/beta. Low T (Low Liquidity) -> High Order.
    # So Low Liquidity -> High Order.
    # High Amihud -> High Order. 
    # Correct.
    
    valid_res = df_res.dropna()
    if len(valid_res) > 0:
        # T ~ 1/Liquidity => q ~ ln(1/T) ~ ln(Illiquidity)
        # Use Log(Amihud) for correlation to match the physics scaling law
        valid_res = valid_res.copy() # Avoid SettingWithCopyWarning
        valid_res['Log_Amihud'] = np.log(valid_res['Amihud_Illiquidity'] + 1e-9)
        
        mech_corr, _ = pearsonr(valid_res['Order_Parameter_M'], valid_res['Log_Amihud'])
        print(f"Mechanism Check: Correlation(Order, Log(Amihud)) = {mech_corr:.4f}")
    
    # 3. Plotting
    print("Plotting Tier 1 Figures...")
    
    # Plot 1: The Triple View (Market, Order Param, Liquidity)
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(14, 12), sharex=True)
    
    # Market
    market_cum = returns.mean(axis=1).cumsum()
    market_cum = market_cum.reindex(df_res.index)
    ax1.plot(market_cum.index, market_cum, color='k', label='S&P 100 Index (Log)')
    ax1.set_ylabel('Market Level')
    ax1.set_title('A. Macroscopic Market State (2000-2024)')
    ax1.grid(True, alpha=0.3)
    
    # Order Param vs Benchmark
    ax2.plot(df_res.index, df_res['Order_Parameter_M'], color='tab:red', label='Parisi Order (Ours)', linewidth=1.5)
    ax2.plot(df_res.index, df_res['Benchmark_Avg_Corr'], color='tab:blue', linestyle='--', label='Avg Correlation (Benchmark)', alpha=0.6)
    ax2.set_ylabel('System Order')
    ax2.set_title('B. Mesoscopic Phase Transition: Our Model vs Standard Benchmark')
    ax2.legend(loc='upper left')
    ax2.grid(True, alpha=0.3)
    
    # Liquidity Mechanism
    # Plot Amihud on log scale potentially, or just normal
    ax3.plot(df_res.index, df_res['Amihud_Illiquidity'], color='tab:green', label='Amihud Illiquidity (Proxy for 1/T)')
    ax3.set_ylabel('Market Illiquidity')
    ax3.set_yscale('log')
    ax3.set_title(f'C. Microscopic Mechanism: Liquidity Driver (Corr with Order: {mech_corr:.2f})')
    ax3.legend(loc='upper left')
    ax3.grid(True, alpha=0.3)
    
    # Highlight Crises
    crises = [
        ('2000-03-01', '2002-10-01', 'Dot-Com'),
        ('2007-10-01', '2009-03-01', 'GFC'),
        ('2020-02-01', '2020-04-01', 'COVID')
    ]
    for start, end, name in crises:
        start_ts, end_ts = pd.Timestamp(start), pd.Timestamp(end)
        for ax in [ax1, ax2, ax3]:
            ax.axvspan(start_ts, end_ts, color='red', alpha=0.1)
            if ax == ax1:
                ax.text(start_ts, ax.get_ylim()[1], name, rotation=0, verticalalignment='bottom')

    plt.tight_layout()
    plt.savefig('figures/tier1_mechanism_proof.png', dpi=300)
    
    # Plot 2: Scatter Mechanism (The "Tier 1" verification plot)
    plt.figure(figsize=(8, 8))
    plt.scatter(valid_res['Amihud_Illiquidity'], valid_res['Order_Parameter_M'], alpha=0.3, c='k', s=10)
    
    # Fit line
    try:
        z = np.polyfit(valid_res['Amihud_Illiquidity'], valid_res['Order_Parameter_M'], 1)
        p = np.poly1d(z)
        plt.plot(valid_res['Amihud_Illiquidity'], p(valid_res['Amihud_Illiquidity']), "r--", label=f'Trend (R={mech_corr:.2f})')
    except:
        pass
        
    plt.xscale('log')
    plt.xlabel('Amihud Illiquidity (Log Scale)')
    plt.ylabel('Parisi Order Parameter (Attention Collapse)')
    plt.title('Mechanism Verification: Liquidity Driven Phase Transition')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig('figures/tier1_scatter_mechanism.png', dpi=300)

    print("Done. Generated Tier 1 validation figures.")

if __name__ == "__main__":
    if not os.path.exists('figures'):
        os.makedirs('figures')
    run_thermodynamics()
