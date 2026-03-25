import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import NMF
from scipy.special import softmax
import os
import warnings

warnings.filterwarnings('ignore')

"""
CHINA A-SHARE BACKTEST (CORRECTED)
----------------------------------
Correction: Prevents "Time Travel" by dynamically handling IPO dates.
Stocks like CATL (300750) are only included in calculation AFTER they list.
"""

def download_china_data_robust():
    tickers = [
        "600519.SS", "601318.SS", "600036.SS", "601166.SS", "600900.SS",
        "600887.SS", "600276.SS", "600030.SS", "601012.SS", "600585.SS",
        "600309.SS", "600048.SS", "601888.SS", "600690.SS", "603288.SS",
        "000858.SZ", "000333.SZ", "000651.SZ", "002415.SZ", "002594.SZ",
        "002714.SZ", "000001.SZ", "000002.SZ", "300015.SZ", "300760.SZ",
        "300750.SZ", "601398.SS", "601288.SS", "601939.SS", "601988.SS"
    ]
    
    start_date = "2010-01-01"
    end_date = "2024-06-01"
    
    print(f"Downloading China Data (2010-2024)...")
    try:
        data = yf.download(tickers, start=start_date, end=end_date, progress=False, auto_adjust=True)
        if isinstance(data.columns, pd.MultiIndex):
            close = data['Close']
        else:
            close = data
            
        # FIX: Do not drop columns based on full history. 
        # Keep all data, including NaNs for pre-IPO periods.
        # Just ensure we remove tickers that failed completely (empty)
        close = close.dropna(axis=1, how='all')
        
        # Forward fill for suspensions, but NOT backward fill (that would be look-ahead)
        close = close.ffill()
        
        return close
    except Exception as e:
        print(f"Error downloading data: {e}")
        return None

class AdiabaticAttentionBacktest:
    def __init__(self, n_components=4): 
        self.n_components = n_components
        
    def fit_step(self, returns_window):
        # FIX: Select only active stocks in this window
        # A stock is "Active" if it has < 30% NaNs in this window
        valid_mask = returns_window.isna().mean() < 0.3
        valid_cols = returns_window.columns[valid_mask]
        
        if len(valid_cols) < 5: return None
        
        clean_window = returns_window[valid_cols].fillna(0)
        
        corr_mat = np.nan_to_num(clean_window.corr().values)
        
        # Adjust components based on available stocks
        n_comp = min(self.n_components, len(valid_cols) // 2)
        if n_comp < 2: n_comp = 2
        
        try:
            model = NMF(n_components=n_comp, init='random', random_state=42, max_iter=20)
            W = model.fit_transform(corr_mat + 1.0)
            H = model.components_
            K = H.T
            
            lambda_reg = 0.5
            K_inv = np.linalg.inv(K.T @ K + lambda_reg * np.eye(n_comp))
            Q_linear = (corr_mat + 1.0) @ K @ K_inv
            
            max_corr = np.max(corr_mat - np.eye(len(corr_mat)))
            T_eff = np.clip(1.0 - max_corr, 0.05, 1.0) 
            
            Q_nonlinear = softmax(Q_linear / T_eff, axis=1)
            return Q_nonlinear
        except:
            return None

def calculate_parisi_order(Q):
    if Q is None: return np.nan
    overlaps = Q @ Q.T
    n = overlaps.shape[0]
    off_diag = overlaps[np.triu_indices(n, k=1)]
    return np.mean(off_diag)

def run_china_backtest():
    close = download_china_data_robust()
    if close is None: return

    # --- 1. Construct Dynamic Index ---
    # Returns of the available universe at each timestep
    returns = close.pct_change()
    
    # Market return is the mean of AVAILABLE stocks on that day
    market_returns = returns.mean(axis=1).fillna(0)
    market_index = (1 + market_returns).cumprod() * 100
    
    # Log returns for model
    log_returns = np.log(close / close.shift(1))
    
    # --- 2. Calculate Signals ---
    window_size = 40 # Slightly longer for China noise
    solver = AdiabaticAttentionBacktest(n_components=4)
    
    print(f"Calculating Signals (Dynamic Universe)...")
    calc_step = 5
    signals = {}
    
    for t in range(window_size, len(log_returns), calc_step):
        if t % 500 == 0: print(f"Processing day {t}...")
        
        current_date = log_returns.index[t]
        window = log_returns.iloc[t-window_size:t]
        
        Q = solver.fit_step(window)
        parisi = calculate_parisi_order(Q)
        signals[current_date] = parisi
        
    sig_series = pd.Series(signals).reindex(market_index.index).ffill()
    
    df_bt = pd.DataFrame({
        'Market_Price': market_index,
        'Market_Ret': market_returns,
        'Parisi': sig_series
    }).dropna()
    
    # --- 3. Walk-Forward Tiered Strategy (China Optimized) ---
    # Adaptation: "Regime-Dependent" Logic for A-Shares.
    # A-Shares allow high crowding in Bull markets (Momentum), 
    # but punish crowding severely in Bear markets (Liquidity Crunch).
    
    lookback = 252
    min_p = 60
    
    roll = df_bt['Parisi'].rolling(window=lookback, min_periods=min_p)
    df_bt['Q80'] = roll.quantile(0.80)
    df_bt['Q95'] = roll.quantile(0.95)
    
    # Fallback
    exp = df_bt['Parisi'].expanding(min_periods=min_p)
    df_bt['Q80'] = df_bt['Q80'].fillna(exp.quantile(0.80))
    df_bt['Q95'] = df_bt['Q95'].fillna(exp.quantile(0.95))
    
    # Trend: Use MA60 (Quarterly Line) as the "Lifeline" to define Bull/Bear regimes
    df_bt['MA60'] = df_bt['Market_Price'].rolling(60).mean()
    
    positions = pd.Series(1.0, index=df_bt.index)
    
    # Regime 1: Bear Market (Price < MA60) -> High Sensitivity
    # In a downtrend, any sign of crowding (Q80) implies a "Trample" event.
    bear_mask = df_bt['Market_Price'] < df_bt['MA60']
    cond_bear_risk = (df_bt['Parisi'] > df_bt['Q80'])
    positions[bear_mask & cond_bear_risk] = 0.0 # Cash is King
    
    # Regime 2: Bull Market (Price > MA60) -> High Tolerance
    # In an uptrend, we tolerate crowding unless it's critical (Q95).
    # Even then, we only reduce to 50% to keep riding the bubble.
    bull_mask = df_bt['Market_Price'] >= df_bt['MA60']
    cond_bull_risk = (df_bt['Parisi'] > df_bt['Q95'])
    positions[bull_mask & cond_bull_risk] = 0.5 
    
    # Regime 3: Physics Limit (Systemic Lock-up)
    # If synchronization > 0.85, liquidity vanishes regardless of trend.
    cond_panic = (df_bt['Parisi'] > 0.85)
    positions[cond_panic] = 0.0
    
    df_bt['Position_Target'] = positions
    df_bt['Real_Position'] = df_bt['Position_Target'].shift(1).fillna(1.0)
    
    # Perf
    RISK_FREE = 0.025 / 252
    COST = 0.0015
    
    df_bt['Strat_Ret'] = df_bt['Real_Position'] * df_bt['Market_Ret'] + \
                         (1 - df_bt['Real_Position']) * RISK_FREE - \
                         df_bt['Real_Position'].diff().abs().fillna(0) * COST
                         
    df_bt['Strat_Wealth'] = (1 + df_bt['Strat_Ret']).cumprod() * 100
    df_bt['Bench_Wealth'] = (1 + df_bt['Market_Ret']).cumprod() * 100
    
    # Stats
    def get_stats(wealth, ret):
        ann_ret = (wealth.iloc[-1]/wealth.iloc[0]) ** (252/len(wealth)) - 1
        ann_vol = ret.std() * np.sqrt(252)
        sharpe = (ann_ret - 0.025) / ann_vol
        peak = wealth.cummax()
        dd = (wealth - peak) / peak
        return ann_ret, ann_vol, sharpe, dd.min()

    b_ret, b_vol, b_sharpe, b_mdd = get_stats(df_bt['Bench_Wealth'], df_bt['Market_Ret'])
    s_ret, s_vol, s_sharpe, s_mdd = get_stats(df_bt['Strat_Wealth'], df_bt['Strat_Ret'])
    
    print("\n" + "="*60)
    print("Strategy: China Optimized (Regime-Dependent: MA60 Bull/Bear Split)")
    print("="*60)
    print(f"{'Metric':<20} | {'Benchmark':<20} | {'Strategy':<20}")
    print("-" * 65)
    print(f"{'Ann. Return':<20} | {b_ret*100:.2f}%{'':<13} | {s_ret*100:.2f}%")
    print(f"{'Ann. Volatility':<20} | {b_vol*100:.2f}%{'':<13} | {s_vol*100:.2f}%")
    print(f"{'Max Drawdown':<20} | {b_mdd*100:.2f}%{'':<13} | {s_mdd*100:.2f}%")
    print(f"{'Sharpe Ratio':<20} | {b_sharpe:.2f}{'':<16} | {s_sharpe:.2f}")
    print("-" * 65)
    
    # Plot
    if not os.path.exists('figures'): os.makedirs('figures')
    plt.figure(figsize=(10,6))
    plt.plot(df_bt.index, df_bt['Bench_Wealth'], 'k--', label='Benchmark')
    plt.plot(df_bt.index, df_bt['Strat_Wealth'], 'r-', label='Strategy')
    plt.title('China A-Share Corrected (No Time Travel)')
    plt.legend()
    plt.savefig('figures/china_market_corrected.png')
    print("Saved figures/china_market_corrected.png")

if __name__ == "__main__":
    run_china_backtest()
