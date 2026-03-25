import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import NMF
from scipy.special import softmax
import os

"""
FIGURE 1: COMPREHENSIVE ORTHOGONALITY CHECK
-------------------------------------------
Panel A: Time-Series Divergence (Parisi vs Linear Systemic Risk)
Panel B: Multi-Factor Orthogonality Heatmap (Parisi vs Vol, Mom, Illiq)
"""

def download_data():
    # Use the same list as core_model.py for consistency
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
    
    print(f"Downloading data for {len(tickers)} tickers...")
    
    # Chunked download logic from core_model.py
    chunk_size = 5
    all_chunks = []
    
    for i in range(0, len(tickers), chunk_size):
        chunk = tickers[i:i+chunk_size]
        try:
            # Force threads=False to sequential download if rate limited
            data_chunk = yf.download(chunk, start=start_date, end=end_date, progress=False, auto_adjust=True, threads=False)
            if not data_chunk.empty:
                all_chunks.append(data_chunk)
        except Exception as e:
            print(f"Chunk failed: {e}")
            
    if not all_chunks:
        print("No data downloaded.")
        return None, None
        
    try:
        data = pd.concat(all_chunks, axis=1)
        data = data.loc[:,~data.columns.duplicated()]

        if isinstance(data.columns, pd.MultiIndex):
            close = data['Close']
            volume = data['Volume']
        else:
            close = data
            volume = None
            
        # STRICT THRESHOLD (0.95) to keep 2000-2024 data
        threshold = int(len(close) * 0.95) 
        close = close.dropna(axis=1, thresh=threshold)
        close = close.ffill().bfill().dropna()
        
        if volume is not None:
             valid_tickers = close.columns
             if isinstance(volume.columns, pd.Index) and not volume.columns.equals(valid_tickers):
                 common = volume.columns.intersection(valid_tickers)
                 volume = volume[common]
             volume = volume.loc[close.index].ffill().bfill()
            
        return close, volume
    except Exception as e:
        print(f"Error processing data: {e}")
        return None, None

class AdiabaticAttentionV2:
    def __init__(self, n_components=5, slow_decay=0.96):
        self.n_components = n_components
        self.slow_decay = slow_decay 
        self.A_slow = None 
        
    def fit_step(self, returns_window):
        if returns_window.shape[1] < 5: return None, 0
        
        corr_mat = returns_window.corr().values
        corr_mat = np.nan_to_num(corr_mat)
        
        if self.A_slow is None:
            self.A_slow = corr_mat
        else:
            self.A_slow = self.slow_decay * self.A_slow + (1 - self.slow_decay) * corr_mat
            
        A_slow_pos = self.A_slow + 1.0
        try:
            model = NMF(n_components=self.n_components, init='random', random_state=42, max_iter=100)
            W = model.fit_transform(A_slow_pos)
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
        return Q_nonlinear, T_eff

def calculate_parisi_order(Q):
    if Q is None: return 0
    overlaps = Q @ Q.T
    n = overlaps.shape[0]
    off_diag = overlaps[np.triu_indices(n, k=1)]
    return np.mean(off_diag)

def run_analysis():
    close, volume = download_data()
    if close is None: return
    
    # 1. Prepare Data
    returns = np.log(close / close.shift(1)).dropna()
    market_index = close.mean(axis=1) # Equal-weighted index
    
    window_size = 60
    step = 5
    solver = AdiabaticAttentionV2(n_components=5)
    
    results = []
    
    print("Running Rolling Analysis...")
    for t in range(window_size, len(returns), step):
        date = returns.index[t]
        window = returns.iloc[t-window_size:t]
        
        # Calculate Parisi
        Q, T_eff = solver.fit_step(window)
        parisi = calculate_parisi_order(Q)
        
        # Calculate Linear Benchmark (Avg Corr)
        avg_corr = window.corr().values[np.triu_indices(window.shape[1], k=1)].mean()
        
        results.append({
            'Date': date,
            'Parisi': parisi,
            'Avg_Corr': avg_corr
        })
        
    df = pd.DataFrame(results).set_index('Date')
    
    # 2. Add Other Factors for Heatmap
    # Align dates
    df.index = pd.to_datetime(df.index)
    
    # A. Volatility (30d rolling std of market index)
    market_ret = market_index.pct_change()
    df['Volatility'] = market_ret.rolling(30).std().reindex(df.index) * np.sqrt(252)
    
    # B. Momentum (12M return)
    df['Momentum'] = market_index.pct_change(250).reindex(df.index)
    
    # C. Amihud Illiquidity (if volume exists)
    if volume is not None:
        # |Ret| / (Price * Volume)
        daily_illiq = (returns.abs() / (close * volume)).mean(axis=1)
        df['Illiquidity'] = daily_illiq.rolling(30).mean().reindex(df.index) * 1e9
    
    df = df.dropna()
    
    # 3. Plotting Figure 1A: Time Series
    if not os.path.exists('JEDC_Submission_Package1/figures'):
        os.makedirs('JEDC_Submission_Package1/figures')
        
    fig1, ax1 = plt.subplots(figsize=(12, 6))
    
    # Dual axis
    lns1 = ax1.plot(df.index, df['Parisi'], 'r-', label='TFI (Non-Linear)')
    ax1.set_ylabel('Topological Fragility Index (TFI)', color='r', fontweight='bold', fontsize=12)
    ax1.tick_params(axis='y', labelcolor='r')
    ax1.set_title('(A) Time-Series Divergence: TFI vs Linear Risk', loc='left', fontsize=14, fontweight='bold')
    
    ax2 = ax1.twinx()
    lns2 = ax2.plot(df.index, df['Avg_Corr'], 'b--', label='Avg Correlation (Linear Proxy)', alpha=0.6)
    ax2.set_ylabel('Average Correlation', color='b', fontweight='bold', fontsize=12)
    ax2.tick_params(axis='y', labelcolor='b')
    
    # Combined legend
    lns = lns1 + lns2
    labs = [l.get_label() for l in lns]
    ax1.legend(lns, labs, loc='upper left', fontsize=10)
    ax1.grid(True, alpha=0.3)
    
    plt.tight_layout()
    save_path_A = 'JEDC_Submission_Package1/figures/Figure_1A_TimeSeries.png'
    plt.savefig(save_path_A, dpi=300)
    print(f"Figure 1A saved to {save_path_A}")

    # 4. Plotting Figure 1B: Heatmap
    fig2, ax3 = plt.subplots(figsize=(10, 8))
    
    # Correlation Matrix
    cols_to_corr = ['Parisi', 'Avg_Corr', 'Volatility', 'Momentum', 'Illiquidity']
    cols_to_corr = [c for c in cols_to_corr if c in df.columns]
    
    # Rename Parisi to TFI for Publication
    df_renamed = df[cols_to_corr].rename(columns={'Parisi': 'TFI'})
    
    corr_mat = df_renamed.corr()
    
    sns.heatmap(corr_mat, annot=True, fmt='.2f', cmap='coolwarm', vmin=-1, vmax=1, center=0, 
                ax=ax3, cbar_kws={'label': 'Pearson Correlation'}, annot_kws={"size": 12})
    ax3.set_title('(B) Multi-Factor Orthogonality Check', loc='left', fontsize=14, fontweight='bold')
    ax3.tick_params(axis='both', which='major', labelsize=12)
    
    plt.tight_layout()
    save_path_B = 'JEDC_Submission_Package1/figures/Figure_1B_Heatmap.png'
    plt.savefig(save_path_B, dpi=300)
    print(f"Figure 1B saved to {save_path_B}")
    
    # Print Correlation Stats for Verification
    print("\nCorrelation Matrix:")
    print(corr_mat)

if __name__ == "__main__":
    run_analysis()
