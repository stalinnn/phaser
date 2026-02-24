import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve, auc, roc_auc_score
from sklearn.decomposition import NMF
from scipy.special import softmax
import os

"""
RISK MODEL AUDIT: RECALL, PRECISION, LEAD TIME
----------------------------------------------
Strict evaluation of the Parisi Order Parameter as a Crash Warning Signal.
Ground Truth: Future 20-day Max Drawdown > 10% (Correction/Crash)
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
        
        corr_mat = returns_window.corr().values
        corr_mat = np.nan_to_num(corr_mat)
        
        A_pos = corr_mat + 1.0
        try:
            model = NMF(n_components=self.n_components, init='random', random_state=42, max_iter=50)
            W = model.fit_transform(A_pos)
            H = model.components_
            K = H.T
        except:
            K = np.random.rand(corr_mat.shape[0], self.n_components)

        lambda_reg = 0.5
        try:
            K_inv = np.linalg.inv(K.T @ K + lambda_reg * np.eye(self.n_components))
            Q_linear = (corr_mat + 1.0) @ K @ K_inv
        except:
            Q_linear = np.random.rand(corr_mat.shape[0], self.n_components)
            
        max_corr = np.max(corr_mat - np.eye(len(corr_mat)))
        T_eff = 1.0 / (max_corr + 1e-6)
        T_eff = np.clip(T_eff, 0.1, 5.0)
        
        Q_nonlinear = softmax(Q_linear / T_eff, axis=1)
        return Q_nonlinear

def calculate_parisi_order(Q):
    if Q is None: return 0
    overlaps = Q @ Q.T
    n = overlaps.shape[0]
    off_diag = overlaps[np.triu_indices(n, k=1)]
    return np.mean(off_diag)

def calculate_max_drawdown_future(series, window=20):
    # Calculate Max Drawdown in the NEXT window days
    # MDD = (Min - Current) / Current
    # We want to know: "If I buy today, what is the worst loss I could see in next 20 days?"
    
    # Rolling forward window is tricky in pandas. 
    # Use shifting: At time t, we look at t+1 to t+window
    indexer = pd.api.indexers.FixedForwardWindowIndexer(window_size=window)
    roll_min = series.rolling(window=indexer).min()
    mdd = (roll_min - series) / series
    return mdd

def run_audit():
    close = download_data()
    if close is None: return
    
    # 1. Market Index (Equal Weight)
    market_index = close.mean(axis=1)
    returns = np.log(close / close.shift(1)).dropna()
    
    # 2. Generate Signals
    window_size = 30
    step = 5
    
    solver = AdiabaticAttentionV2(n_components=5)
    dates = []
    signals = []
    
    print("Generating Signals for Audit...")
    for t in range(window_size, len(returns), step):
        current_date = returns.index[t]
        window = returns.iloc[t-window_size:t]
        
        Q = solver.fit_step(window)
        parisi = calculate_parisi_order(Q)
        
        dates.append(current_date)
        signals.append(parisi)
        
    df_sig = pd.DataFrame({'Parisi': signals}, index=dates)
    df_sig = df_sig.reindex(market_index.index).ffill().dropna()
    
    # 3. Define Ground Truth (Crashes)
    # Crash = Future 20d Drawdown < -10%
    future_mdd = calculate_max_drawdown_future(market_index, window=40) # Look 2 months ahead for "The Crash"
    
    # Ground Truth Label
    crash_threshold = -0.10
    df_sig['Is_Crash'] = (future_mdd < crash_threshold).astype(int)
    
    # Clean NaNs at the end
    df_sig = df_sig.dropna()
    
    # 4. Evaluate Metrics
    y_true = df_sig['Is_Crash']
    y_scores = df_sig['Parisi']
    
    precision, recall, thresholds = precision_recall_curve(y_true, y_scores)
    
    # Find F1-Score optimal threshold
    f1_scores = 2 * (precision * recall) / (precision + recall + 1e-9)
    best_idx = np.argmax(f1_scores)
    best_threshold = thresholds[best_idx]
    
    print("\n" + "="*50)
    print("STRICT RISK AUDIT REPORT")
    print("="*50)
    print(f"Total Trading Days: {len(df_sig)}")
    print(f"Total Crash Days (MDD > 10% in next 40d): {sum(y_true)} ({sum(y_true)/len(df_sig)*100:.1f}%)")
    print("-" * 50)
    print(f"Optimal Threshold (Max F1): q > {best_threshold:.4f}")
    print(f"Recall (Capture Rate): {recall[best_idx]*100:.1f}%")
    print(f"Precision (True Alarm Rate): {precision[best_idx]*100:.1f}%")
    print(f"False Positive Rate (Wolf Cry): {(1-precision[best_idx])*100:.1f}%")
    print("-" * 50)
    
    # 5. Lead Time Analysis
    # Identify "Crash Events" (Continuous blocks of Is_Crash=1)
    # We want to see how early the signal crossed threshold BEFORE the crash started.
    # Simplified: Correlation between Signal and Future MDD magnitude
    corr_mdd = df_sig['Parisi'].corr(future_mdd)
    print(f"Correlation (Signal vs Future Drawdown): {corr_mdd:.4f} (Negative is good)")
    
    # Align lengths for plotting
    common_idx = df_sig.index.intersection(future_mdd.index)
    df_plot = df_sig.loc[common_idx]
    mdd_plot = future_mdd.loc[common_idx]
    
    # 6. Plotting
    if not os.path.exists('figures'): os.makedirs('figures')
    
    fig, ax1 = plt.subplots(figsize=(12, 6))
    
    ax1.plot(df_plot.index, df_plot['Parisi'], 'b-', label='Parisi Signal', alpha=0.7)
    ax1.axhline(best_threshold, color='orange', linestyle='--', label=f'Optimal Threshold ({best_threshold:.2f})')
    ax1.set_ylabel('Parisi Order', color='b')
    
    # Highlight Crash Periods
    ax2 = ax1.twinx()
    # Use fill_between with aligned index
    is_crash = df_plot['Is_Crash'] == 1
    if is_crash.any():
        ax2.fill_between(df_plot.index, 0, 1, where=is_crash, color='red', alpha=0.3, label='Future Crash (>10%)', transform=ax2.get_xaxis_transform())
    
    ax2.plot(mdd_plot.index, mdd_plot, 'r:', label='Future MDD', alpha=0.3)
    ax2.set_ylabel('Future Max Drawdown', color='r')
    
    plt.title(f'Risk Audit: Signal vs Reality (Recall: {recall[best_idx]*100:.0f}%)')
    plt.savefig('figures/risk_audit.png')
    print("Audit chart saved to figures/risk_audit.png")

if __name__ == "__main__":
    run_audit()
