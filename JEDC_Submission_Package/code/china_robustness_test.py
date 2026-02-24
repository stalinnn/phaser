import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import NMF
from scipy.special import softmax
import os

"""
CHINA MARKET ROBUSTNESS & ORTHOGONALITY CHECK
---------------------------------------------
Objective:
1. Orthogonality: Correlation between Parisi and Historical Volatility in China.
2. Robustness: Heatmap of Sharpe Ratio across K (Components) and W (Window).
"""

def download_china_data():
    tickers = [
        "600519.SS", "601318.SS", "600036.SS", "601166.SS", "600900.SS",
        "000858.SZ", "000333.SZ", "000651.SZ", "002415.SZ", "002594.SZ",
        "601398.SS", "601288.SS", "601939.SS", "601988.SS"
    ]
    start_date = "2010-01-01"
    end_date = "2024-06-01"
    
    print(f"Downloading China Data...")
    try:
        data = yf.download(tickers, start=start_date, end=end_date, progress=False, auto_adjust=True)
        if isinstance(data.columns, pd.MultiIndex):
            close = data['Close']
        else:
            close = data
        close = close.ffill().bfill()
        threshold = int(len(close) * 0.6) 
        close = close.dropna(axis=1, thresh=threshold).dropna()
        return close
    except:
        return None

class AdiabaticAttentionLite:
    def __init__(self, n_components=4):
        self.n_components = n_components
        
    def get_parisi(self, returns_window, T_scale=1.0):
        if returns_window.shape[1] < 5: return np.nan
        corr_mat = np.nan_to_num(returns_window.corr().values)
        
        try:
            model = NMF(n_components=self.n_components, init='random', random_state=42, max_iter=10) # Faster
            W = model.fit_transform(corr_mat + 1.0)
            H = model.components_
            K = H.T
            
            lambda_reg = 0.5
            K_inv = np.linalg.inv(K.T @ K + lambda_reg * np.eye(self.n_components))
            Q_linear = (corr_mat + 1.0) @ K @ K_inv
            
            max_corr = np.max(corr_mat - np.eye(len(corr_mat)))
            T_eff = np.clip(1.0 - max_corr, 0.05, 1.0) * T_scale
            
            Q_nonlinear = softmax(Q_linear / T_eff, axis=1)
            overlaps = Q_nonlinear @ Q_nonlinear.T
            n = overlaps.shape[0]
            off_diag = overlaps[np.triu_indices(n, k=1)]
            return np.mean(off_diag)
        except:
            return np.nan

def run_tests():
    close = download_china_data()
    if close is None: return
    
    # Pre-calc returns
    market_index = close.mean(axis=1)
    market_ret = market_index.pct_change().fillna(0)
    log_ret = np.log(close / close.shift(1)).dropna()
    
    # ---------------------------------------------------------
    # TEST 1: Orthogonality (Parisi vs Volatility)
    # ---------------------------------------------------------
    print("Running Orthogonality Test...")
    solver = AdiabaticAttentionLite(n_components=4)
    window = 30
    
    parisi_list = []
    vol_list = []
    dates = []
    
    # Calculate rolling stats
    for t in range(window, len(log_ret), 5):
        w_ret = log_ret.iloc[t-window:t]
        p = solver.get_parisi(w_ret)
        v = market_ret.iloc[t-window:t].std() * np.sqrt(252) # Ann. Vol
        
        if not np.isnan(p):
            parisi_list.append(p)
            vol_list.append(v)
            dates.append(log_ret.index[t])
            
    df_orth = pd.DataFrame({'Parisi': parisi_list, 'Volatility': vol_list}, index=dates)
    corr = df_orth.corr().iloc[0,1]
    
    print(f"Correlation (Parisi vs Vol): {corr:.4f}")
    
    # Plot Orthogonality
    if not os.path.exists('figures'): os.makedirs('figures')
    
    fig, ax1 = plt.subplots(figsize=(10, 6))
    ax1.scatter(df_orth['Volatility'], df_orth['Parisi'], alpha=0.2, s=10, color='purple')
    ax1.set_xlabel('Historical Volatility (Ann.)')
    ax1.set_ylabel('Parisi Order Parameter')
    ax1.set_title(f'China Market Orthogonality Check\nCorrelation = {corr:.2f} (Low correlation implies unique information)')
    ax1.grid(True, alpha=0.3)
    plt.savefig('figures/china_orthogonality.png')
    
    # ---------------------------------------------------------
    # TEST 2: Parameter Robustness (Heatmap)
    # ---------------------------------------------------------
    print("Running Parameter Robustness Scan (This may take a while)...")
    
    K_range = [3, 4, 5, 6, 7]
    W_range = [20, 30, 45, 60, 90]
    
    sharpe_matrix = np.zeros((len(K_range), len(W_range)))
    
    for i, K in enumerate(K_range):
        for j, W in enumerate(W_range):
            print(f"Testing K={K}, W={W}...")
            solver = AdiabaticAttentionLite(n_components=K)
            
            # Fast signal gen
            sigs = []
            valid_rets = market_ret.iloc[W:]
            step = 5
            
            # Iterate through time
            # Note: Aligning indices carefully
            for t in range(0, len(valid_rets), step):
                # Map back to log_ret index
                # log_ret index t corresponds to valid_rets t-W? No.
                # Let's just grab window from log_ret
                idx = t + W
                if idx >= len(log_ret): break
                
                w_ret = log_ret.iloc[idx-W:idx]
                p = solver.get_parisi(w_ret)
                sigs.append(p)
            
            # Simple Backtest
            # Signal: Parisi > 0.4 -> Cash
            # Align signal to returns (ffill)
            s_series = pd.Series(sigs, index=log_ret.index[W::step][:len(sigs)])
            s_daily = s_series.reindex(market_ret.index).ffill().dropna()
            
            # Align market ret
            m_daily = market_ret.loc[s_daily.index]
            
            # Logic: If Parisi > 0.4, Pos=0
            # Shift 1 day
            pos = (s_daily <= 0.4).astype(int).shift(1).fillna(1)
            
            strat_ret = pos * m_daily
            
            # Calc Sharpe
            ann_ret = strat_ret.mean() * 252
            ann_vol = strat_ret.std() * np.sqrt(252)
            sharpe = (ann_ret - 0.025) / (ann_vol + 1e-6)
            
            sharpe_matrix[i, j] = sharpe
            
    # Plot Heatmap
    plt.figure(figsize=(8, 6))
    sns.heatmap(sharpe_matrix, annot=True, fmt=".2f", 
                xticklabels=W_range, yticklabels=K_range, cmap="RdYlGn")
    plt.xlabel("Window Size (Days)")
    plt.ylabel("Latent Components (K)")
    plt.title("China Strategy Robustness (Sharpe Ratio)")
    plt.savefig('figures/china_robustness.png')
    
    print("Done. Saved to figures/china_orthogonality.png and figures/china_robustness.png")

if __name__ == "__main__":
    run_tests()
