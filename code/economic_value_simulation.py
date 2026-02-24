import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import NMF
from scipy.special import softmax
import os

"""
ECONOMIC VALUE SIMULATION: THE $1 BILLION FUND
----------------------------------------------
Scenario: A $1B Fund tracks S&P 100.
Strategy: "The Smart Sentinel"
    - Signal: Parisi > 0.4 (Crowding)
    - Filter: Price < MA(50) (Trend Weak)
    - Action: Buy Put Option (30x Leverage)
    - Exit: Stop after 5 days if no crash (Time Stop)
Goal: Prove positive CAGR over 24 years.
"""

def download_data():
    tickers = [
        "MSFT", "AMZN", "JPM", "XOM", "GE", "JNJ", "PFE", "C", "KO", "PEP",
        "WMT", "MRK", "INTC", "CSCO", "IBM", "GS", "BAC", "AIG", "AXP", "MCD",
        "AAPL", "GOOGL", "NVDA", "TSLA", "META", "BRK-B", "UNH", "LLY", "V", "PG",
        "HD", "MA", "CVX", "ABBV", "ADBE", "NFLX", "DIS", "CMCSA", "TXN", "PM", 
        "HON", "QCOM", "AMGN", "CAT", "SPGI", "MS", "BA", "MMM", "T", "VZ"
    ]
    tickers = list(set(tickers))
    
    start_date = "2000-01-01" 
    end_date = "2024-01-01"
    
    print(f"Downloading data (2000-2024)...")
    try:
        # Download Stocks with threads=False to avoid rate limits
        data = yf.download(tickers, start=start_date, end=end_date, progress=False, auto_adjust=True, threads=False)
        if isinstance(data.columns, pd.MultiIndex):
            close = data['Close']
        else:
            close = data
            
        # Download VIX Separately
        try:
            vix_data = yf.download("^VIX", start=start_date, end=end_date, progress=False, auto_adjust=True, threads=False)
            if isinstance(vix_data.columns, pd.MultiIndex):
                vix = vix_data['Close']['^VIX'] 
            elif 'Close' in vix_data.columns:
                vix = vix_data['Close']
            else:
                vix = vix_data
            if len(vix) < 10: raise ValueError("VIX empty")
        except:
            print("VIX download failed, using dummy VIX=20")
            vix = pd.Series(20, index=close.index) 
            
        close = close.dropna(axis=1, thresh=int(len(close)*0.7)).ffill().bfill().dropna()
        vix = vix.reindex(close.index).ffill().bfill()
        
        if close.empty: raise ValueError("Stock data empty")
        
        return close, vix
    except Exception as e:
        print(f"Download failed: {e}. Generating SYNTHETIC data for demonstration.")
        # Synthetic Data Fallback
        dates = pd.date_range(start=start_date, end=end_date, freq='B')
        n = len(dates)
        # S&P 100 Proxy: Drift + Shock
        np.random.seed(42)
        returns = np.random.normal(0.0003, 0.01, (n, 50)) # 50 stocks
        
        # Add Crises
        # 2008
        idx_2008 = np.where((dates.year==2008) & (dates.month>=9))[0]
        returns[idx_2008] -= 0.02 # Heavy drag
        # 2020
        idx_2020 = np.where((dates.year==2020) & (dates.month==3))[0]
        returns[idx_2020] -= 0.05 # Flash crash
        
        close_synth = pd.DataFrame(100 * np.cumprod(1 + returns, axis=0), index=dates, columns=[f"S{i}" for i in range(50)])
        
        # Synthetic VIX (inverse to market)
        market = close_synth.mean(axis=1)
        roll_vol = market.pct_change().rolling(20).std() * np.sqrt(252) * 100
        vix_synth = roll_vol.fillna(20) + 10 # Base VIX
        
        return close_synth, vix_synth

class AdiabaticAttentionV2:
    def __init__(self, n_components=5):
        self.n_components = n_components
        
    def fit_step(self, returns_window):
        if returns_window.shape[1] < 5: return None
        corr_mat = np.nan_to_num(returns_window.corr().values)
        A_pos = corr_mat + 1.0
        try:
            model = NMF(n_components=self.n_components, init='random', random_state=42, max_iter=20)
            W = model.fit_transform(A_pos)
            H = model.components_
            K = H.T
        except:
            K = np.random.rand(corr_mat.shape[0], self.n_components)

        try:
            lambda_reg = 0.5
            K_inv = np.linalg.inv(K.T @ K + lambda_reg * np.eye(self.n_components))
            Q_linear = (corr_mat + 1.0) @ K @ K_inv
        except:
            Q_linear = np.random.rand(corr_mat.shape[0], self.n_components)
            
        max_corr = np.max(corr_mat - np.eye(len(corr_mat)))
        T_eff = np.clip(1.0 / (max_corr + 1e-6), 0.1, 5.0)
        
        Q_nonlinear = softmax(Q_linear / T_eff, axis=1)
        return Q_nonlinear

def calculate_parisi_order(Q):
    if Q is None: return 0
    overlaps = Q @ Q.T
    n = overlaps.shape[0]
    off_diag = overlaps[np.triu_indices(n, k=1)]
    return np.mean(off_diag)

def run_economic_simulation():
    data_tuple = download_data()
    if data_tuple is None: return
    close, vix = data_tuple
    
    # 1. Market Index
    market_index = close.mean(axis=1)
    returns = np.log(close / close.shift(1)).dropna()
    
    # 2. Generate Signal
    window_size = 30
    step = 5
    solver = AdiabaticAttentionV2(n_components=5)
    
    dates = []
    signals = []
    
    print("Generating Risk Signals...")
    for t in range(window_size, len(returns), step):
        current_date = returns.index[t]
        window = returns.iloc[t-window_size:t]
        Q = solver.fit_step(window)
        parisi = calculate_parisi_order(Q)
        dates.append(current_date)
        signals.append(parisi)
        
    df = pd.DataFrame({'Parisi': signals}, index=dates)
    df = df.reindex(market_index.index).ffill().dropna()
    
    # Align Data
    aligned_vix = vix.loc[df.index]
    market_price = (1 + market_index.pct_change()).cumprod()
    ma_50 = market_price.rolling(50).mean()
    aligned_ma = ma_50.loc[df.index]
    aligned_price = market_price.loc[df.index]
    aligned_ret = market_index.pct_change().loc[df.index]
    
    # 3. Simulate $1B Fund Performance
    initial_aum = 1_000_000_000 # $1 Billion
    
    # Strategy Parameters: THE "SMART SENTINEL"
    parisi_threshold = 0.40
    
    # Cost Management: Buy 5-day put. If no crash, expire.
    # Cost = 0.25% per trade (approx premium for 5-day OTM put)
    per_trade_cost = 0.0025 
    leverage = 30.0 
    
    nav_benchmark = [initial_aum]
    nav_hedged = [initial_aum]
    
    current_nav_b = initial_aum
    current_nav_h = initial_aum
    
    days_since_hedge = 999
    is_hedged = False
    
    print("Simulating Daily P&L (Sentinel Strategy)...")
    
    for date, ret in aligned_ret.items():
        if pd.isna(ret): continue
        
        current_nav_b *= (1 + ret)
        nav_benchmark.append(current_nav_b)
        
        try:
            signal = df.loc[:date].iloc[-2]['Parisi']
            price = aligned_price.loc[:date].iloc[-2]
            ma = aligned_ma.loc[:date].iloc[-2]
        except:
            signal = 0
            price = 100
            ma = 100
            
        # ENTRY LOGIC:
        # 1. Crowding High (Parisi > 0.4)
        # 2. Trend Weak (Price < MA 50) -- Crucial filter!
        condition_met = (signal > parisi_threshold) and (price < ma)
        
        # EXPIRE LOGIC:
        if is_hedged and days_since_hedge >= 5:
            is_hedged = False # Expired
            
        # New Entry?
        if condition_met and not is_hedged:
            is_hedged = True
            days_since_hedge = 0
            # Pay cost immediately upon entry
            current_nav_h *= (1 - per_trade_cost)
            
        equity_pnl = current_nav_h * ret
        option_pnl = 0
        
        if is_hedged:
            days_since_hedge += 1
            if ret < 0:
                # Gamma gains
                exposure = current_nav_h * per_trade_cost * leverage
                payoff = exposure * max(0, -ret)
                option_pnl += payoff
        
        current_nav_h += (equity_pnl + option_pnl)
        nav_hedged.append(current_nav_h)
        
    # --- ANALYSIS ---
    nav_b_series = pd.Series(nav_benchmark, index=[df.index[0]] + df.index.tolist())
    nav_h_series = pd.Series(nav_hedged, index=[df.index[0]] + df.index.tolist())
    
    days = (nav_b_series.index[-1] - nav_b_series.index[0]).days
    years = days / 365.25
    
    cagr_b = (nav_b_series.iloc[-1] / initial_aum) ** (1/years) - 1
    cagr_h = (nav_h_series.iloc[-1] / initial_aum) ** (1/years) - 1
    
    vol_b = nav_b_series.pct_change().std() * np.sqrt(252)
    vol_h = nav_h_series.pct_change().std() * np.sqrt(252)
    
    dd_b = (nav_b_series - nav_b_series.cummax()) / nav_b_series.cummax()
    mdd_b = dd_b.min()
    
    dd_h = (nav_h_series - nav_h_series.cummax()) / nav_h_series.cummax()
    mdd_h = dd_h.min()
    
    sharpe_b = cagr_b / vol_b
    sharpe_h = cagr_h / vol_h
    
    print("\n" + "="*60)
    print(f"FULL CYCLE PERFORMANCE (2000-2024)")
    print("="*60)
    print(f"{'Metric':<15} | {'Benchmark (S&P 100)':<20} | {'Hedged (Smart Sentinel)':<20}")
    print("-" * 65)
    print(f"{'CAGR':<15} | {cagr_b*100:.2f}%{'':<14} | {cagr_h*100:.2f}%")
    print(f"{'Volatility':<15} | {vol_b*100:.2f}%{'':<14} | {vol_h*100:.2f}%")
    print(f"{'Max Drawdown':<15} | {mdd_b*100:.2f}%{'':<14} | {mdd_h*100:.2f}%")
    print(f"{'Sharpe Ratio':<15} | {sharpe_b:.2f}{'':<16} | {sharpe_h:.2f}")
    print("-" * 65)
    
    # Crises Analysis
    crises = [
        ("2000 DotCom", "2000-03-01", "2002-10-01"),
        ("2008 GFC", "2008-09-01", "2009-03-01"),
        ("2020 Covid", "2020-02-01", "2020-04-01")
    ]
    
    print("\n" + "="*60)
    print("CRISIS SPECIFIC PERFORMANCE")
    print("="*60)
    
    for name, start, end in crises:
        try:
            val_start_b = nav_b_series.loc[start:].iloc[0]
            val_end_b = nav_b_series.loc[:end].iloc[-1]
            loss_b = val_start_b - val_end_b
            val_start_h = nav_h_series.loc[start:].iloc[0]
            val_end_h = nav_h_series.loc[:end].iloc[-1]
            loss_h = val_start_h - val_end_h
            saved = loss_b - loss_h
            
            print(f"--- {name} ---")
            print(f"Benchmark Loss: -${loss_b/1e6:.1f} M  ({(val_end_b/val_start_b-1)*100:.1f}%)")
            print(f"Hedged Strategy Loss: -${loss_h/1e6:.1f} M  ({(val_end_h/val_start_h-1)*100:.1f}%)")
            print(f"MONEY SAVED: ${saved/1e6:.1f} MILLION")
            print("-" * 30)
        except:
            pass

    if not os.path.exists('figures'): os.makedirs('figures')
    plt.figure(figsize=(12, 6))
    plt.plot(nav_b_series, 'k--', label='Benchmark ($1B)', alpha=0.6)
    plt.plot(nav_h_series, 'g-', label='Hedged Portfolio (Sentinel)', linewidth=1.5)
    plt.title('The Smart Sentinel Strategy: Full Cycle Performance')
    plt.ylabel('Portfolio Value ($)')
    plt.yscale('log')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig('figures/economic_value.png')
    print("\nChart saved to figures/economic_value.png")

if __name__ == "__main__":
    run_economic_simulation()
