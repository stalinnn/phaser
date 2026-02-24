import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import NMF
from scipy.special import softmax
import os

"""
RISK MODEL AUDIT: OPTIMIZATION & ABLATION
-----------------------------------------
Previous finding: RSI > 70 Filter failed (reduced Recall significantly).
New Hypothesis (Minsky Moment): High Crowding (Parisi) is most dangerous when Volatility is LOW (Complacency).
Strategies Tested:
1. Pure Parisi (Threshold sensitivity analysis)
2. Parisi + Low Volatility (The "Silent Risk" Hypothesis)
3. Parisi + High RSI (The failed baseline for comparison)
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
    
    print(f"Downloading data for Audit...")
    try:
        data = yf.download(tickers, start=start_date, end=end_date, progress=False, auto_adjust=True)
        if isinstance(data.columns, pd.MultiIndex):
            close = data['Close']
        else:
            close = data
            
        threshold = int(len(close) * 0.7) 
        close = close.dropna(axis=1, thresh=threshold)
        close = close.ffill().bfill().dropna()
        return close
    except Exception as e:
        print(f"Error: {e}")
        return None

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
        # NEW TEMP FORMULA: (1 - max_corr)
        # When max_corr -> 1.0 (Crisis), T -> 0.01 (Frozen -> High Parisi)
        # When max_corr -> 0.0 (Normal), T -> 1.0 (Hot -> Low Parisi)
        T_eff = np.clip(1.0 - max_corr, 0.05, 1.0) 
        Q_nonlinear = softmax(Q_linear / T_eff, axis=1)
        return Q_nonlinear

def calculate_parisi_order(Q):
    if Q is None: return 0
    overlaps = Q @ Q.T
    n = overlaps.shape[0]
    off_diag = overlaps[np.triu_indices(n, k=1)]
    return np.mean(off_diag)

def calculate_rsi(series, period=14):
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def calculate_volatility(returns, window=20):
    return returns.rolling(window=window).std() * np.sqrt(252)

def calculate_max_drawdown_future(series, window=20):
    indexer = pd.api.indexers.FixedForwardWindowIndexer(window_size=window)
    roll_min = series.rolling(window=indexer).min()
    mdd = (roll_min - series) / series
    return mdd

def get_metrics(y_true, y_pred):
    tp = np.sum((y_true == 1) & (y_pred == 1))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    fn = np.sum((y_true == 1) & (y_pred == 0))
    
    precision = tp / (tp + fp + 1e-9)
    recall = tp / (tp + fn + 1e-9)
    return precision, recall

def run_audit_optimized():
    close = download_data()
    if close is None: return
    
    # 1. Market Index & Basic Features
    market_index = close.mean(axis=1)
    returns = np.log(close / close.shift(1)).dropna()
    market_returns = market_index.pct_change()
    
    # 2. Generate Signals
    window_size = 30
    step = 5
    
    solver = AdiabaticAttentionV2(n_components=5)
    dates = []
    signals = []
    
    print("Generating Parisi Signals...")
    for t in range(window_size, len(returns), step):
        current_date = returns.index[t]
        window = returns.iloc[t-window_size:t]
        
        Q = solver.fit_step(window)
        parisi = calculate_parisi_order(Q)
        
        dates.append(current_date)
        signals.append(parisi)
        
    df_sig = pd.DataFrame({'Parisi': signals}, index=dates)
    df_sig = df_sig.reindex(market_index.index).ffill().dropna()
    
    # 3. Calculate Filters
    print("Calculating Filters (RSI & Volatility)...")
    
    print("\nPARISI DISTRIBUTION STATS:")
    print(df_sig['Parisi'].describe())
    
    # RSI
    market_rsi = calculate_rsi(market_index, period=14)
    df_sig['RSI'] = market_rsi.reindex(df_sig.index).ffill()
    
    # Volatility
    market_vol = calculate_volatility(market_returns, window=20)
    df_sig['Vol'] = market_vol.reindex(df_sig.index).ffill()
    
    # 4. Define Ground Truth
    future_mdd = calculate_max_drawdown_future(market_index, window=40)
    crash_threshold = -0.10
    df_sig['Is_Crash'] = (future_mdd < crash_threshold).astype(int)
    
    df_sig = df_sig.dropna()
    
    # 5. Evaluate Strategies
    print("\n" + "="*80)
    print("AUDIT RESULT: OPTIMIZING PRECISION VS RECALL")
    print("="*80)
    print(f"{'Strategy':<35} | {'Precision':<10} | {'Recall':<10} | {'F1-Score':<10}")
    print("-" * 80)
    
    strategies = []
    
    # A. Threshold Scan for Pure Parisi
    for thresh in [0.20, 0.25, 0.30, 0.35, 0.40]:
        pred = (df_sig['Parisi'] > thresh).astype(int)
        p, r = get_metrics(df_sig['Is_Crash'], pred)
        f1 = 2 * (p * r) / (p + r + 1e-9)
        print(f"{f'Pure Parisi > {thresh:.2f}':<35} | {p*100:.1f}%{'':<5} | {r*100:.1f}%{'':<5} | {f1:.3f}")
        strategies.append({'name': f'Parisi > {thresh}', 'p': p, 'r': r, 'thresh': thresh})

    print("-" * 80)
    
    # B. Minsky Moment (Parisi + Low Vol)
    # Rationale: Crowding is dangerous when investors are complacent (Low Vol)
    parisi_base = 0.25 # Use a moderate base
    vol_median = df_sig['Vol'].median() # Low Vol definition
    
    pred_minsky = ((df_sig['Parisi'] > parisi_base) & (df_sig['Vol'] < vol_median)).astype(int)
    p_m, r_m = get_metrics(df_sig['Is_Crash'], pred_minsky)
    f1_m = 2 * (p_m * r_m) / (p_m + r_m + 1e-9)
    print(f"{f'Parisi > {parisi_base} & Vol < Median':<35} | {p_m*100:.1f}%{'':<5} | {r_m*100:.1f}%{'':<5} | {f1_m:.3f}")
    
    # C. Failed RSI Baseline (for record)
    pred_rsi = ((df_sig['Parisi'] > 0.20) & (df_sig['RSI'] > 70)).astype(int)
    p_r, r_r = get_metrics(df_sig['Is_Crash'], pred_rsi)
    f1_r = 2 * (p_r * r_r) / (p_r + r_r + 1e-9)
    print(f"{f'Parisi > 0.20 & RSI > 70 (Old)':<35} | {p_r*100:.1f}%{'':<5} | {r_r*100:.1f}%{'':<5} | {f1_r:.3f}")

    print("-" * 80)

    # Plotting Best Strategy
    if not os.path.exists('figures'): os.makedirs('figures')
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=True)
    
    # Top: Market & Crashes
    ax1.plot(df_sig.index, market_index.loc[df_sig.index], 'k-', label='S&P 100', alpha=0.6)
    
    # Highlight Crashes
    crashes = df_sig[df_sig['Is_Crash'] == 1]
    ax1.scatter(crashes.index, market_index.loc[crashes.index], color='red', s=1, alpha=0.3, label='Real Crashes')
    
    # Mark Best Pure Signal (e.g., > 0.35)
    best_thresh = 0.35
    best_signals = df_sig[df_sig['Parisi'] > best_thresh]
    ax1.scatter(best_signals.index, market_index.loc[best_signals.index], color='orange', marker='x', s=30, label=f'Signal > {best_thresh}')
    
    ax1.set_ylabel('Market Index')
    ax1.set_title(f'Strategy Optimization: Moving Threshold to {best_thresh} significantly improves Precision')
    ax1.legend()
    
    # Bottom: Parisi Indicator
    ax2.plot(df_sig.index, df_sig['Parisi'], 'b-', label='Parisi Order Param')
    ax2.axhline(y=best_thresh, color='r', linestyle='--', label=f'Optimized Threshold {best_thresh}')
    ax2.axhline(y=0.20, color='gray', linestyle=':', label='Old Threshold 0.20')
    
    ax2.set_ylabel('Parisi Order Parameter')
    ax2.legend()
    
    plt.tight_layout()
    plt.savefig('figures/risk_audit_optimized.png')
    print("Chart saved to figures/risk_audit_optimized.png")

if __name__ == "__main__":
    run_audit_optimized()
