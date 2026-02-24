import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import NMF
from scipy.special import softmax
import os

"""
THRESHOLD SENSITIVITY ANALYSIS: METRICS SCAN
--------------------------------------------
Metrics: Precision, Recall, Lead Time across varying Parisi Thresholds.
Markets: US vs China.
Crash Definition: Future 20-day Drawdown > 10%.
"""

def get_data(market='US'):
    if market == 'US':
        tickers = ["MSFT", "AMZN", "JPM", "XOM", "GE", "JNJ", "PFE", "C", "KO", "PEP"]
        start = "2000-01-01"
    else:
        tickers = ["600519.SS", "601318.SS", "600036.SS", "601166.SS", "600900.SS"]
        start = "2010-01-01"
        
    try:
        data = yf.download(tickers, start=start, end="2024-06-01", progress=False, auto_adjust=True)
        if isinstance(data.columns, pd.MultiIndex):
            close = data['Close']
        else:
            close = data
        return close.ffill().bfill().dropna(axis=1)
    except:
        return None

class AdiabaticAttention:
    def __init__(self, n_components=4):
        self.n_components = n_components
        
    def get_parisi(self, returns_window):
        if returns_window.shape[1] < 3: return 0
        corr_mat = np.nan_to_num(returns_window.corr().values)
        try:
            model = NMF(n_components=self.n_components, init='random', random_state=42, max_iter=10)
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

def calc_metrics(market_name):
    print(f"\nScanning {market_name} Market...")
    close = get_data(market_name)
    if close is None: return None
    
    market_index = close.mean(axis=1)
    returns = np.log(close / close.shift(1)).dropna()
    
    # Calc Parisi
    solver = AdiabaticAttention()
    parisi = []
    window = 30
    dates = []
    for t in range(window, len(returns), 5):
        w = returns.iloc[t-window:t]
        p = solver.get_parisi(w)
        dates.append(returns.index[t])
        parisi.append(p)
        
    df = pd.DataFrame({'Parisi': parisi}, index=dates)
    df = df.reindex(market_index.index).ffill()
    
    # Define Ground Truth (Future Crash)
    # Look ahead 20 days: if max drawdown > 10%, then it's a crash period
    indexer = pd.api.indexers.FixedForwardWindowIndexer(window_size=20)
    future_min = market_index.rolling(window=indexer).min()
    future_dd = (future_min - market_index) / market_index
    df['Is_Crash'] = (future_dd < -0.10).astype(int)
    
    # Scan Thresholds
    thresholds = np.arange(0.2, 0.85, 0.05)
    results = []
    
    for thresh in thresholds:
        # Signal: Parisi > Threshold
        pred = (df['Parisi'] > thresh).astype(int)
        
        # Precision: TP / (TP + FP)
        # Of all alarms, how many were followed by a crash?
        tp = np.sum((pred == 1) & (df['Is_Crash'] == 1))
        fp = np.sum((pred == 1) & (df['Is_Crash'] == 0))
        precision = tp / (tp + fp + 1e-9)
        
        # Recall: TP / (TP + FN)
        # Of all crash days, how many were flagged?
        fn = np.sum((pred == 0) & (df['Is_Crash'] == 1))
        recall = tp / (tp + fn + 1e-9)
        
        # Lead Time Calculation (Only for True Positives)
        # We define lead time as (Crash_Start_Date - First_Signal_Date)
        # But signals can be noisy. Let's look at the *first* signal in the 60-day window before crash.
        
        lead_times = []
        for crash_idx in np.where(df['Is_Crash'] == 1)[0]:
            # This is a day marked as "Crash Imminent" (Future return is bad)
            # But we want the *start* of the crash event to measure back from.
            # This logic is tricky on daily time series.
            pass
            
        # Alternative Lead Time Logic:
        # 1. Identify Crash Start Dates (Peak before drop > 10%)
        # 2. For each crash, look back 60 days.
        # 3. Find first day where Parisi > Threshold.
        # 4. Lead = Crash_Date - Signal_Date
        
        # Re-use logic from calculate_lead_time_stats.py but integrated here
        # Need to re-identify crash starts
        rolling_max = market_index.rolling(250, min_periods=1).max()
        drawdown = (market_index - rolling_max) / rolling_max
        is_crash_start = (drawdown < -0.10) & (drawdown.shift(1) >= -0.10) # Crossing -10% line? No, that's late.
        
        # Better: Local peaks followed by -10% drop within 20 days
        # Let's use the 'Is_Crash' flag we already built.
        # Is_Crash=1 means "If you buy today, you lose >10% in next 20 days".
        # So the "Crash Start" is roughly the first day Is_Crash flips from 0 to 1.
        
        crash_flip = (df['Is_Crash'] == 1) & (df['Is_Crash'].shift(1) == 0)
        crash_dates = df.index[crash_flip]
        
        current_leads = []
        for c_date in crash_dates:
            # Look back 60 days
            window_start = c_date - pd.Timedelta(days=60)
            sub_df = df.loc[window_start:c_date]
            
            # First signal
            sig_dates = sub_df[sub_df['Parisi'] > thresh].index
            if len(sig_dates) > 0:
                first_sig = sig_dates[0]
                lead = (c_date - first_sig).days
                if lead > 0: current_leads.append(lead)
        
        avg_lead = np.mean(current_leads) if len(current_leads) > 0 else 0
        
        results.append({
            'Threshold': thresh,
            'Precision': precision,
            'Recall': recall,
            'F1': 2 * (precision * recall) / (precision + recall + 1e-9),
            'Avg Lead Time (Days)': avg_lead
        })
        
    return pd.DataFrame(results)

def run_analysis():
    res_us = calc_metrics('US')
    res_cn = calc_metrics('China')
    
    # Print Tables
    if res_us is not None:
        print("\n--- US Market Metrics ---")
        print(res_us.round(3))
        
    if res_cn is not None:
        print("\n--- China Market Metrics ---")
        print(res_cn.round(3))
        
    # Plotting
    if not os.path.exists('figures'): os.makedirs('figures')
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # US Plot
    if res_us is not None:
        ax1.set_xlabel('TFI Threshold')
        ax1.set_ylabel('Score')
        l1, = ax1.plot(res_us['Threshold'], res_us['Precision'], 'b-o', label='Precision')
        l2, = ax1.plot(res_us['Threshold'], res_us['Recall'], 'r-s', label='Recall')
        
        # Twin axis for Lead Time
        ax1_twin = ax1.twinx()
        l3, = ax1_twin.plot(res_us['Threshold'], res_us['Avg Lead Time (Days)'], 'g-^', label='Lead Time', linestyle=':')
        ax1_twin.set_ylabel('Lead Time (Days)', color='g')
        ax1_twin.tick_params(axis='y', labelcolor='g')
        
        ax1.set_title('US Market: Metrics vs Threshold')
        ax1.legend([l1, l2, l3], ['Precision', 'Recall', 'Lead Time'], loc='center right')
        ax1.grid(True, alpha=0.3)
        
    # China Plot
    if res_cn is not None:
        ax2.set_xlabel('TFI Threshold')
        ax2.set_ylabel('Score')
        l1, = ax2.plot(res_cn['Threshold'], res_cn['Precision'], 'b-o', label='Precision')
        l2, = ax2.plot(res_cn['Threshold'], res_cn['Recall'], 'r-s', label='Recall')
        
        # Twin axis for Lead Time
        ax2_twin = ax2.twinx()
        l3, = ax2_twin.plot(res_cn['Threshold'], res_cn['Avg Lead Time (Days)'], 'g-^', label='Lead Time', linestyle=':')
        ax2_twin.set_ylabel('Lead Time (Days)', color='g')
        ax2_twin.tick_params(axis='y', labelcolor='g')
        
        ax2.set_title('China Market: Metrics vs Threshold')
        ax2.legend([l1, l2, l3], ['Precision', 'Recall', 'Lead Time'], loc='center right')
        ax2.grid(True, alpha=0.3)
        
    plt.tight_layout()
    plt.savefig('figures/threshold_sensitivity.png')
    print("\nSaved chart to figures/threshold_sensitivity.png")

if __name__ == "__main__":
    run_analysis()
