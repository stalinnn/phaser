import yfinance as yf
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.decomposition import NMF
from scipy.special import softmax
import os

"""
MULTI-FACTOR ORTHOGONALITY CHECK
--------------------------------
Objective: Verify if Parisi contains unique information compared to:
1. Volatility (Risk)
2. Momentum (Trend)
3. Amihud Illiquidity (Market Friction)
"""

def get_data():
    tickers = [
        "600519.SS", "601318.SS", "600036.SS", "601166.SS", "600900.SS",
        "000858.SZ", "000333.SZ", "000651.SZ", "002415.SZ", "002594.SZ"
    ]
    start = "2015-01-01"
    end = "2024-01-01"
    
    print("Downloading Data...")
    try:
        data = yf.download(tickers, start=start, end=end, progress=False, auto_adjust=True)
        close = data['Close'].ffill().bfill()
        volume = data['Volume'].ffill().bfill()
        # Ensure volume is not zero
        volume = volume.replace(0, np.nan).fillna(method='ffill')
        return close, volume
    except:
        return None, None

class AdiabaticAttention:
    def __init__(self, n_components=4):
        self.n_components = n_components
        
    def get_parisi(self, returns_window):
        if returns_window.shape[1] < 3: return np.nan
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
            return np.nan

def run_check():
    close, volume = get_data()
    if close is None: return
    
    # 1. Calculate Factors
    market_index = close.mean(axis=1)
    returns = np.log(close / close.shift(1))
    
    # Factor A: Parisi
    solver = AdiabaticAttention()
    parisi_list = []
    dates = []
    window = 30
    
    print("Calculating Parisi...")
    for t in range(window, len(returns), 5):
        w = returns.iloc[t-window:t].dropna()
        p = solver.get_parisi(w)
        parisi_list.append(p)
        dates.append(returns.index[t])
        
    df = pd.DataFrame({'Parisi': parisi_list}, index=dates)
    
    # Factor B: Volatility (30d Std)
    df['Volatility'] = market_index.pct_change().rolling(30).std().reindex(df.index)
    
    # Factor C: Momentum (12M Return)
    df['Momentum'] = market_index.pct_change(250).reindex(df.index)
    
    # Factor D: Amihud Illiquidity
    # |Ret| / (Volume * Price)
    daily_illip = (returns.abs() / (volume * close)).mean(axis=1)
    # Moving average of market-wide illiquidity
    df['Illiquidity'] = daily_illip.rolling(30).mean().reindex(df.index)
    
    # Clean
    df = df.dropna()
    
    # 2. Correlation Matrix
    corr_matrix = df.corr()
    print("\n--- Correlation Matrix ---")
    print(corr_matrix.round(3))
    
    # 3. Plot
    if not os.path.exists('figures'): os.makedirs('figures')
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1, center=0)
    plt.title('Orthogonality Check: Parisi vs Traditional Factors')
    plt.savefig('figures/multi_orthogonality.png')
    print("Saved chart to figures/multi_orthogonality.png")

if __name__ == "__main__":
    run_check()
