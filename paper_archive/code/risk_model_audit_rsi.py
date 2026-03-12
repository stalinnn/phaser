import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve
from sklearn.decomposition import NMF
from scipy.special import softmax
import os

"""
RISK MODEL AUDIT: PARISI + RSI FILTER
-------------------------------------
Hypothesis: High Crowding (Parisi) is only dangerous when combined with Overbought conditions (RSI > 70).
Goal: Improve Precision (reduce false alarms) while maintaining acceptable Recall.
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
        T_eff = np.clip(1.0 / (max_corr + 1e-6), 0.1, 5.0)
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

def calculate_max_drawdown_future(series, window=20):
    indexer = pd.api.indexers.FixedForwardWindowIndexer(window_size=window)
    roll_min = series.rolling(window=indexer).min()
    mdd = (roll_min - series) / series
    return mdd

def run_audit_rsi():
    close = download_data()
    if close is None: return
    
    # 1. Market Index
    market_index = close.mean(axis=1)
    returns = np.log(close / close.shift(1)).dropna()
    
    # 2. Generate Signals
    window_size = 30
    step = 5
    
    solver = AdiabaticAttentionV2(n_components=5)
    dates = []
    signals = []
    
    print("Generating Signals...")
    for t in range(window_size, len(returns), step):
        current_date = returns.index[t]
        window = returns.iloc[t-window_size:t]
        
        Q = solver.fit_step(window)
        parisi = calculate_parisi_order(Q)
        
        dates.append(current_date)
        signals.append(parisi)
        
    df_sig = pd.DataFrame({'Parisi': signals}, index=dates)
    df_sig = df_sig.reindex(market_index.index).ffill().dropna()
    
    # 3. Calculate RSI
    print("Calculating RSI...")
    market_rsi = calculate_rsi(market_index, period=14)
    df_sig['RSI'] = market_rsi.reindex(df_sig.index).ffill()
    
    # 4. Define Ground Truth
    future_mdd = calculate_max_drawdown_future(market_index, window=40)
    crash_threshold = -0.10
    df_sig['Is_Crash'] = (future_mdd < crash_threshold).astype(int)
    
    df_sig = df_sig.dropna()
    
    # 5. Evaluate: Simple Parisi vs Dual Signal
    
    # Thresholds
    parisi_thresh = 0.20 # From previous optimal
    rsi_thresh = 70      # Standard Overbought
    
    # Logic 1: Parisi Only
    pred_parisi = (df_sig['Parisi'] > parisi_thresh).astype(int)
    
    # Logic 2: Parisi + RSI (Dual Confirmation)
    # We relax Parisi threshold slightly because we have RSI confirmation? 
    # Let's keep 0.20 first.
    pred_dual = ((df_sig['Parisi'] > parisi_thresh) & (df_sig['RSI'] > rsi_thresh)).astype(int)
    
    def get_metrics(y_true, y_pred, name):
        tp = np.sum((y_true == 1) & (y_pred == 1))
        fp = np.sum((y_true == 0) & (y_pred == 1))
        fn = np.sum((y_true == 1) & (y_pred == 0))
        
        precision = tp / (tp + fp + 1e-9)
        recall = tp / (tp + fn + 1e-9)
        return precision, recall
    
    p1, r1 = get_metrics(df_sig['Is_Crash'], pred_parisi, "Parisi Only")
    p2, r2 = get_metrics(df_sig['Is_Crash'], pred_dual, "Dual (Parisi+RSI)")
    
    print("\n" + "="*50)
    print("AUDIT: RSI FILTER IMPACT")
    print("="*50)
    print(f"{'Metric':<15} | {'Parisi Only (>0.2)':<20} | {'Dual (Parisi>0.2 & RSI>70)':<25}")
    print("-" * 65)
    print(f"{'Precision':<15} | {p1*100:.1f}%{'':<14} | {p2*100:.1f}%  (Target: Higher)")
    print(f"{'Recall':<15} | {r1*100:.1f}%{'':<14} | {r2*100:.1f}%  (Target: Stable)")
    print("-" * 65)
    
    # Plotting
    if not os.path.exists('figures'): os.makedirs('figures')
    
    fig, ax1 = plt.subplots(figsize=(12, 6))
    
    ax1.plot(df_sig.index, market_index.loc[df_sig.index], 'k-', label='S&P 100', alpha=0.5)
    ax1.set_ylabel('Market Index')
    
    # Mark Dual Signals
    crash_signals = df_sig[pred_dual == 1]
    ax1.scatter(crash_signals.index, market_index.loc[crash_signals.index], color='red', s=10, label='Dual Signal')
    
    plt.title(f'Dual Signal Precision: {p2*100:.1f}% (vs Baseline {p1*100:.1f}%)')
    plt.legend()
    plt.savefig('figures/risk_audit_rsi.png')
    print("Chart saved to figures/risk_audit_rsi.png")

if __name__ == "__main__":
    run_audit_rsi()
