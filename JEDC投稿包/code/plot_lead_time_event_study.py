import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import NMF
from scipy.special import softmax
import os

"""
EVENT STUDY: DEMONSTRATING LEAD TIME
------------------------------------
Focus: The 2020 COVID-19 Crash (Feb-Mar 2020).
Objective: Show that Parisi peaked BEFORE the market bottomed out.
"""

def get_event_data():
    tickers = [
        "MSFT", "AMZN", "JPM", "XOM", "GE", "JNJ", "PFE", "C", "KO", "PEP",
        "WMT", "MRK", "INTC", "CSCO", "IBM", "GS", "BAC", "AIG", "AXP", "MCD",
        "AAPL", "GOOGL", "NVDA", "TSLA", "META", "BRK-B", "UNH", "LLY", "V", "PG"
    ]
    # Zoom in on the crisis window
    start = "2019-10-01"
    end = "2020-06-01"
    
    data = yf.download(tickers, start=start, end=end, progress=False, auto_adjust=True)['Close']
    data = data.ffill().bfill().dropna(axis=1)
    return data

class AdiabaticAttention:
    def __init__(self, n_components=5):
        self.n_components = n_components
        
    def get_tfi(self, returns_window):
        if returns_window.shape[1] < 5: return 0
        corr_mat = np.nan_to_num(returns_window.corr().values)
        try:
            model = NMF(n_components=self.n_components, init='random', random_state=42, max_iter=20)
            W = model.fit_transform(corr_mat + 1.0)
            H = model.components_
            K = H.T
            
            lambda_reg = 0.5
            K_inv = np.linalg.inv(K.T @ K + lambda_reg * np.eye(self.n_components))
            Q_linear = (corr_mat + 1.0) @ K @ K_inv
            
            max_corr = np.max(corr_mat - np.eye(len(corr_mat)))
            T_eff = np.clip(1.0 - max_corr, 0.05, 1.0) 
            
            Q_nonlinear = softmax(Q_linear / T_eff, axis=1)
            overlaps = Q_nonlinear @ Q_nonlinear.T
            n = overlaps.shape[0]
            off_diag = overlaps[np.triu_indices(n, k=1)]
            return np.mean(off_diag)
        except:
            return 0

def run_event_study():
    close = get_event_data()
    market_index = close.mean(axis=1)
    returns = np.log(close / close.shift(1)).dropna()
    
    solver = AdiabaticAttention(n_components=5)
    tfi_vals = []
    dates = []
    
    # Use a rolling window of 30 days, step 1 day for high resolution
    window_size = 30
    
    print("Calculating TFI for 2020 Event...")
    for t in range(window_size, len(returns)):
        w = returns.iloc[t-window_size:t]
        p = solver.get_tfi(w)
        dates.append(returns.index[t])
        tfi_vals.append(p)
        
    df = pd.DataFrame({'TFI': tfi_vals}, index=dates)
    df = df.reindex(market_index.index).ffill()
    
    # Plotting
    if not os.path.exists('figures'): os.makedirs('figures')
    
    fig, ax1 = plt.subplots(figsize=(12, 6))
    
    # Market Price (Black)
    color = 'black'
    ax1.set_xlabel('Date')
    ax1.set_ylabel('S&P 100 Proxy Index', color=color)
    ax1.plot(df.index, market_index, color=color, linewidth=2, label='Market Price')
    ax1.tick_params(axis='y', labelcolor=color)
    
    # Highlight the Crash
    # Peak: Feb 19, 2020. Bottom: Mar 23, 2020.
    crash_start = pd.Timestamp('2020-02-19')
    crash_bottom = pd.Timestamp('2020-03-23')
    
    # TFI Signal (Red)
    ax2 = ax1.twinx()
    color = 'tab:red'
    ax2.set_ylabel('Topological Fragility Index (TFI)', color=color)
    ax2.plot(df.index, df['TFI'], color=color, linewidth=1.5, linestyle='--', label='TFI Signal')
    ax2.tick_params(axis='y', labelcolor=color)
    ax2.set_ylim(0, 1.0)
    
    # Add Critical Threshold line
    threshold = 0.40
    ax2.axhline(threshold, color='red', linestyle=':', alpha=0.5)
    
    # Find when TFI crossed threshold
    # Look for the first crossing in Jan/Feb 2020
    crossing_date = df[(df.index > '2020-01-01') & (df['TFI'] > threshold)].index[0]
    print(f"Signal Date: {crossing_date}")
    print(f"Crash Date: {crash_start}")
    lead_days = (crash_start - crossing_date).days
    
    # Annotate
    plt.title(f'Event Study: 2020 COVID-19 Crash\nSignal Lead Time: ~{lead_days} Days before Market Peak', fontweight='bold')
    
    # Draw Arrows
    ax2.annotate('Signal Triggered\n(Crowding Detected)', xy=(crossing_date, threshold), xytext=(crossing_date, 0.8),
                 arrowprops=dict(facecolor='red', shrink=0.05), color='red', ha='center')
                 
    ax1.annotate('Market Peak\n(Crash Begins)', xy=(crash_start, market_index.loc[crash_start]), xytext=(crash_start+pd.Timedelta(days=15), market_index.loc[crash_start]+10),
                 arrowprops=dict(facecolor='black', shrink=0.05), color='black')

    # Shade the "Lead Time" zone
    ax1.axvspan(crossing_date, crash_start, color='yellow', alpha=0.2, label='Lead Time Window')
    
    plt.savefig('figures/event_study_2020.png', dpi=300, bbox_inches='tight')
    print("Saved to figures/event_study_2020.png")

if __name__ == "__main__":
    run_event_study()
