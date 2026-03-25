import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import NMF
from scipy.special import softmax
import os
import warnings
import data_provider as dp  # NEW: Import the data provider

warnings.filterwarnings('ignore')

"""
FULL BACKTEST: ADIABATIC ATTENTION STRATEGY (WALK-FORWARD EDITION)
------------------------------------------------------------------
Updated to use data_provider for robust data loading.
"""

class AdiabaticAttentionBacktest:
    def __init__(self, n_components=5):
        self.n_components = n_components
        
    def fit_step(self, returns_window):
        # 1. Dynamic Universe Handling
        # Critical for historical data: Only select stocks active in this window
        # Drop columns with any NaNs in this specific window to ensure matrix integrity
        valid_window = returns_window.dropna(axis=1, how='any')
        
        if valid_window.shape[1] < 10: 
            return None # Need at least 10 stocks for meaningful topology
        
        # Use valid_window for correlation
        corr_mat = np.nan_to_num(valid_window.corr().values)
        
        # 2. Adiabatic NMF
        A_pos = corr_mat + 1.0
        try:
            # Dynamic component reduction if universe is small
            n_comp = min(self.n_components, valid_window.shape[1] // 2)
            model = NMF(n_components=n_comp, init='random', random_state=42, max_iter=50)
            W = model.fit_transform(A_pos)
            H = model.components_
            K = H.T
        except:
            return None

        # 3. Attention Mechanism
        try:
            lambda_reg = 0.5
            K_inv = np.linalg.inv(K.T @ K + lambda_reg * np.eye(n_comp))
            Q_linear = (corr_mat + 1.0) @ K @ K_inv
        except:
            return None
            
        # 4. Optimized Temperature
        max_corr = np.max(corr_mat - np.eye(len(corr_mat)))
        T_eff = np.clip(1.0 - max_corr, 0.05, 1.0) 
        
        Q_nonlinear = softmax(Q_linear / T_eff, axis=1)
        return Q_nonlinear

def calculate_tfi(Q):
    if Q is None: return np.nan
    overlaps = Q @ Q.T
    n = overlaps.shape[0]
    off_diag = overlaps[np.triu_indices(n, k=1)]
    return np.mean(off_diag)

def run_backtest():
    # --- 1. Load Data via Provider ---
    # Will try to load 'data/market_data_cleaned.csv' first, then fallback to yfinance
    close = dp.load_market_data(start_date="2000-01-01", end_date="2024-01-01")
    
    if close is None or close.empty:
        print("Data load failed.")
        return

    # --- 2. Construct Market Index (Dynamic Universe) ---
    # Construct an equal-weight index of WHATEVER is available that day
    # returns = ln(Pt / Pt-1)
    returns = np.log(close / close.shift(1))
    
    # Mean return of all available stocks for that day (Index Proxy)
    market_returns = returns.mean(axis=1).fillna(0)
    market_index = (1 + market_returns).cumprod() * 100
    
    # --- 3. Calculate Signal History (Walk Forward) ---
    window_size = 60 
    
    solver = AdiabaticAttentionBacktest(n_components=5)
    
    print(f"Calculating TFI Signals (Walk-Forward) on {close.shape[1]} assets...")
    
    calc_step = 5 
    signals = {}
    
    # Start loop
    for t in range(window_size, len(returns), calc_step):
        if t % 500 == 0: print(f"Processing day {t}/{len(returns)}...")
        
        current_date = returns.index[t]
        
        # Slice window
        window = returns.iloc[t-window_size:t]
        
        # Calculate Logic
        Q = solver.fit_step(window)
        tfi = calculate_tfi(Q)
        
        signals[current_date] = tfi
        
    # Create Signal Series
    sig_series = pd.Series(signals).reindex(market_index.index).ffill()
    
    df_bt = pd.DataFrame({
        'Market_Price': market_index,
        'Market_Ret': market_returns,
        'TFI': sig_series
    }).dropna()
    
    # --- 4. Walk-Forward Tiered Strategy ---
    lookback_days = 252
    min_periods = 60
    
    # Dynamic Percentiles (Rolling Window to avoid Look-ahead)
    roll = df_bt['TFI'].rolling(window=lookback_days, min_periods=min_periods)
    df_bt['Q80'] = roll.quantile(0.80) 
    df_bt['Q95'] = roll.quantile(0.95)
    
    # Fallback for start
    exp = df_bt['TFI'].expanding(min_periods=min_periods)
    df_bt['Q80'] = df_bt['Q80'].fillna(exp.quantile(0.80))
    df_bt['Q95'] = df_bt['Q95'].fillna(exp.quantile(0.95))
    
    # Trend Filter
    df_bt['MA20'] = df_bt['Market_Price'].rolling(window=20).mean()
    
    # Signal Logic
    cond_panic = (df_bt['TFI'] > 0.75)
    cond_critical = (df_bt['TFI'] > df_bt['Q95'])
    cond_caution = (df_bt['TFI'] > df_bt['Q80']) & (df_bt['Market_Price'] < df_bt['MA20'])
    
    positions = pd.Series(1.0, index=df_bt.index)
    positions[cond_caution] = 0.5 
    positions[cond_critical | cond_panic] = 0.0
    
    df_bt['Position_Target'] = positions
    df_bt['Real_Position'] = df_bt['Position_Target'].shift(1).fillna(1.0)
    
    # Performance Stats
    RISK_FREE_RATE_DAILY = 0.02 / 252
    TRANSACTION_COST = 0.001 # 10bps
    
    df_bt['Strat_Ret_Gross'] = df_bt['Real_Position'] * df_bt['Market_Ret'] + (1 - df_bt['Real_Position']) * RISK_FREE_RATE_DAILY
    df_bt['Cost'] = df_bt['Real_Position'].diff().abs().fillna(0) * TRANSACTION_COST
    df_bt['Strat_Ret_Net'] = df_bt['Strat_Ret_Gross'] - df_bt['Cost']
    
    df_bt['Strat_Wealth'] = (1 + df_bt['Strat_Ret_Net']).cumprod() * 100
    df_bt['Bench_Wealth'] = (1 + df_bt['Market_Ret']).cumprod() * 100
    
    def get_stats(wealth, ret):
        if len(wealth) == 0: return 0,0,0,0
        ann_ret = (wealth.iloc[-1]/wealth.iloc[0]) ** (252/len(wealth)) - 1
        ann_vol = ret.std() * np.sqrt(252)
        sharpe = (ann_ret - 0.02) / (ann_vol + 1e-9)
        dd = (wealth / wealth.cummax()) - 1
        return ann_ret, ann_vol, sharpe, dd.min()

    b_ret, b_vol, b_sharpe, b_mdd = get_stats(df_bt['Bench_Wealth'], df_bt['Market_Ret'])
    s_ret, s_vol, s_sharpe, s_mdd = get_stats(df_bt['Strat_Wealth'], df_bt['Strat_Ret_Net'])
    
    print("\n" + "="*60)
    print("WALK-FORWARD BACKTEST RESULTS (Enhanced Data)")
    print("="*60)
    print(f"{'Metric':<20} | {'Benchmark':<20} | {'Strategy (WF)':<20}")
    print("-" * 65)
    print(f"{'Ann. Return':<20} | {b_ret*100:.2f}%{'':<13} | {s_ret*100:.2f}%")
    print(f"{'Ann. Volatility':<20} | {b_vol*100:.2f}%{'':<13} | {s_vol*100:.2f}%")
    print(f"{'Max Drawdown':<20} | {b_mdd*100:.2f}%{'':<13} | {s_mdd*100:.2f}%")
    print(f"{'Sharpe Ratio':<20} | {b_sharpe:.2f}{'':<16} | {s_sharpe:.2f}")
    print("-" * 65)

    # Plot
    if not os.path.exists('figures'): os.makedirs('figures')
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=True)
    
    ax1.plot(df_bt.index, df_bt['Bench_Wealth'], 'k-', label='Benchmark (Equal Weight)', alpha=0.6)
    ax1.plot(df_bt.index, df_bt['Strat_Wealth'], 'r-', label='Strategy (TFI Adjusted)')
    ax1.set_yscale('log')
    ax1.set_title('Cumulative Wealth (Survivorship-Adjusted Universe Proxy)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    ax2.plot(df_bt.index, df_bt['TFI'], 'b-', alpha=0.5, label='Topological Fragility Index (TFI)')
    ax2.plot(df_bt.index, df_bt['Q95'], 'r--', label='Critical (95%)')
    ax2.plot(df_bt.index, df_bt['Q80'], 'orange', linestyle=':', label='Warning (80%)', alpha=0.7)
    ax2.fill_between(df_bt.index, 0, 1, where=(df_bt['Real_Position'] < 1.0), color='gray', alpha=0.2, label='De-risking')
    ax2.set_ylabel('Crowding Metric (TFI)')
    ax2.legend(loc='upper left')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('figures/backtest_enhanced_data.png')
    # Also save as fig3 for paper compatibility
    plt.savefig('figures/fig3_tarc_performance.png')
    print("Saved Figure: figures/backtest_enhanced_data.png and figures/fig3_tarc_performance.png")

if __name__ == "__main__":
    run_backtest()
