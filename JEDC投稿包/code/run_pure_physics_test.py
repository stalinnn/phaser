import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import NMF
from scipy.special import softmax
import os

"""
PURE PHYSICS TEST: RAW PARISI SIGNAL (NO FILTERS)
-------------------------------------------------
Objective: Test the raw predictive power of the Parisi Order Parameter across US and China markets.
Strategy:
  - If Parisi > 0.40: CASH (Risk Off)
  - If Parisi <= 0.40: MARKET (Risk On)
  - No Trend Filter, No Smoothing, No RSI. Just pure topology.
"""

def download_data(market='US'):
    if market == 'US':
        tickers = [
            "MSFT", "AMZN", "JPM", "XOM", "GE", "JNJ", "PFE", "C", "KO", "PEP",
            "WMT", "MRK", "INTC", "CSCO", "IBM", "GS", "BAC", "AIG", "AXP", "MCD",
            "AAPL", "GOOGL", "NVDA", "TSLA", "META", "BRK-B", "UNH", "LLY", "V", "PG"
        ]
        start_date = "2000-01-01"
    else:
        # China A-Shares (Blue Chips)
        tickers = [
            "600519.SS", "601318.SS", "600036.SS", "601166.SS", "600900.SS",
            "000858.SZ", "000333.SZ", "000651.SZ", "002415.SZ", "002594.SZ",
            "601398.SS", "601288.SS", "601939.SS", "601988.SS"
        ]
        start_date = "2010-01-01"
        
    end_date = "2024-06-01"
    print(f"Downloading {market} Data...")
    try:
        data = yf.download(tickers, start=start_date, end=end_date, progress=False, auto_adjust=True)
        if isinstance(data.columns, pd.MultiIndex):
            close = data['Close']
        else:
            close = data
        
        # Clean
        close = close.ffill().bfill()
        threshold = int(len(close) * 0.6) 
        close = close.dropna(axis=1, thresh=threshold)
        close = close.dropna()
        return close
    except:
        return None

class AdiabaticAttention:
    def __init__(self, n_components=4):
        self.n_components = n_components
        
    def get_parisi(self, returns_window):
        if returns_window.shape[1] < 5: return 0
        corr_mat = np.nan_to_num(returns_window.corr().values)
        
        # 1. NMF
        try:
            model = NMF(n_components=self.n_components, init='random', random_state=42, max_iter=20)
            W = model.fit_transform(corr_mat + 1.0)
            H = model.components_
            K = H.T
        except:
            return 0

        # 2. Attention
        try:
            lambda_reg = 0.5
            K_inv = np.linalg.inv(K.T @ K + lambda_reg * np.eye(self.n_components))
            Q_linear = (corr_mat + 1.0) @ K @ K_inv
        except:
            return 0
            
        # 3. Temp
        max_corr = np.max(corr_mat - np.eye(len(corr_mat)))
        T_eff = np.clip(1.0 - max_corr, 0.05, 1.0) 
        Q_nonlinear = softmax(Q_linear / T_eff, axis=1)
        
        # 4. Parisi
        overlaps = Q_nonlinear @ Q_nonlinear.T
        n = overlaps.shape[0]
        off_diag = overlaps[np.triu_indices(n, k=1)]
        return np.mean(off_diag)

def backtest_market(market_name):
    close = download_data(market_name)
    if close is None: return None
    
    market_index = close.mean(axis=1)
    market_index = market_index / market_index.iloc[0] * 100
    returns = np.log(close / close.shift(1)).dropna()
    market_ret = market_index.pct_change().fillna(0)
    
    solver = AdiabaticAttention()
    parisi_vals = []
    dates = []
    
    window_size = 30
    step = 5
    
    for t in range(window_size, len(returns), step):
        w = returns.iloc[t-window_size:t]
        p = solver.get_parisi(w)
        dates.append(returns.index[t])
        parisi_vals.append(p)
        
    df = pd.DataFrame({'Parisi': parisi_vals}, index=dates)
    df = df.reindex(market_index.index).ffill().dropna()
    df['Market_Ret'] = market_ret
    
    # Pure Strategy: Parisi > 0.4 -> Cash
    df['Signal'] = (df['Parisi'] > 0.40).astype(int)
    # Shift 1 day to avoid look-ahead
    df['Position'] = 1 - df['Signal'].shift(1).fillna(0)
    
    # Returns
    df['Strat_Ret'] = df['Position'] * df['Market_Ret']
    
    df['Bench_Wealth'] = (1 + df['Market_Ret']).cumprod() * 100
    df['Strat_Wealth'] = (1 + df['Strat_Ret']).cumprod() * 100
    
    # Stats
    def get_max_dd(wealth):
        dd = (wealth - wealth.cummax()) / wealth.cummax()
        return dd.min()
        
    ann_ret = (df['Strat_Wealth'].iloc[-1]/100) ** (252/len(df)) - 1
    mdd = get_max_dd(df['Strat_Wealth'])
    bench_mdd = get_max_dd(df['Bench_Wealth'])
    
    return df, ann_ret, mdd, bench_mdd

def run_comparison():
    print("Running US Market...")
    res_us = backtest_market('US')
    
    print("Running China Market...")
    res_cn = backtest_market('China')
    
    if not os.path.exists('figures'): os.makedirs('figures')
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Plot US
    if res_us:
        df, ret, mdd, b_mdd = res_us
        ax1.plot(df.index, df['Bench_Wealth'], 'k--', label='Benchmark (S&P 100)', alpha=0.5)
        ax1.plot(df.index, df['Strat_Wealth'], 'b-', label='Pure Parisi Strategy', linewidth=2)
        ax1.set_yscale('log')
        ax1.set_title(f"US Market (2000-2024)\nMax DD: {mdd*100:.1f}% (vs {b_mdd*100:.1f}%)")
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
    # Plot China
    if res_cn:
        df, ret, mdd, b_mdd = res_cn
        ax2.plot(df.index, df['Bench_Wealth'], 'k--', label='Benchmark (China Blue Chips)', alpha=0.5)
        ax2.plot(df.index, df['Strat_Wealth'], 'r-', label='Pure Parisi Strategy', linewidth=2)
        ax2.set_yscale('log')
        ax2.set_title(f"China Market (2010-2024)\nMax DD: {mdd*100:.1f}% (vs {b_mdd*100:.1f}%)")
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
    plt.tight_layout()
    plt.savefig('figures/pure_physics_test.png')
    print("Saved to figures/pure_physics_test.png")

if __name__ == "__main__":
    run_comparison()
