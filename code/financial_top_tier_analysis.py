
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import NMF, PCA
from sklearn.linear_model import LinearRegression
from scipy.stats import pearsonr
import statsmodels.api as sm
import os

"""
TIER 1 QUANTITATIVE FINANCE ANALYSIS
------------------------------------
Goal: Prove Incremental Predictive Power over PCA and Simple Correlation.
1. Benchmark 1: Rolling Mean Correlation
2. Benchmark 2: PCA First Component Variance (PC1) - The standard measure of Systemic Risk (Billio et al.)
3. Incremental Information Test: Does Attention-Order predict Volatility better than PC1?
"""

def download_data():
    # S&P 100 Approximation (Top liquid stocks)
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
    try:
        data = yf.download(tickers, start=start_date, end=end_date, progress=False, auto_adjust=True)
        
        if isinstance(data.columns, pd.MultiIndex):
            close = data['Close']
        else:
            close = data
            
        # Clean Data
        # Drop columns with too many NaNs (late listing)
        threshold = int(len(close) * 0.7) 
        close = close.dropna(axis=1, thresh=threshold)
        
        # Fill remaining
        close = close.ffill().bfill().dropna()
        
        print(f"Data Shape: {close.shape}")
        return close
    except Exception as e:
        print(f"Error: {e}")
        return None

from scipy.stats import skew, kurtosis, pearsonr

class AdiabaticAttention:
    def __init__(self, n_components=3, slow_decay=0.98):
        self.n_components = n_components
        self.slow_decay = slow_decay 
        self.A_slow = None 
        self.K_current = None
        
    def fit_step(self, returns_window):
        if returns_window.shape[1] < 5: return None
        
        # 1. Instantaneous Correlation (Fast Manifold)
        corr_mat = returns_window.corr().values
        corr_mat = np.nan_to_num(corr_mat)
        
        # 2. Slow Manifold Update
        if self.A_slow is None:
            self.A_slow = corr_mat
        else:
            self.A_slow = self.slow_decay * self.A_slow + (1 - self.slow_decay) * corr_mat
            
        # 3. Extract Keys (Fundamental Attributes) via NMF on Slow Manifold
        A_slow_pos = self.A_slow + 1.0
        try:
            model = NMF(n_components=self.n_components, init='random', random_state=42, max_iter=100)
            W = model.fit_transform(A_slow_pos)
            H = model.components_
            self.K_current = H.T 
        except:
            self.K_current = np.random.rand(corr_mat.shape[0], self.n_components)

        # 4. Solve for Query (Investor Attention) - WITH NON-LINEARITY
        # Instead of linear inversion, we posit that Investor Attention Q drives the correlation structure.
        # But to capture "Panic", we need to apply the Softmax bottleneck.
        
        # First, get the linear estimate (Latent factors)
        lambda_reg = 0.5
        K = self.K_current
        try:
            K_inv = np.linalg.inv(K.T @ K + lambda_reg * np.eye(self.n_components))
            Q_linear = (corr_mat + 1.0) @ K @ K_inv
        except:
            Q_linear = np.random.rand(corr_mat.shape[0], self.n_components)
            
        # CRITICAL UPGRADE: Apply Temperature-dependent Softmax
        # We assume the "Effective Temperature" is related to Market Volatility (Inverse)
        # High Vol -> Low Temp -> Sharp Attention (Tunnel Vision)
        # Low Vol -> High Temp -> Diffuse Attention
        
        # Estimate intrinsic noise (Temperature)
        # If market is calm, T is high (1.0). If panic, T -> 0.
        market_vol = np.mean(np.diag(corr_mat)) # Proxy, or use returns std
        # Actually, let's use the inverse of the max correlation as a proxy for T
        max_corr = np.max(corr_mat - np.eye(len(corr_mat)))
        T_eff = 1.0 / (max_corr + 1e-6) 
        T_eff = np.clip(T_eff, 0.1, 5.0)
        
        # Apply Softmax along the component dimension
        # This simulates "Resource Constraint" - investors can only pay attention to limited factors
        from scipy.special import softmax
        Q_nonlinear = softmax(Q_linear / T_eff, axis=1)
        
        return Q_nonlinear

def calculate_parisi_order(Q):
    if Q is None: return 0
    # For probability distributions (after softmax), using Jensen-Shannon or plain Dot Product
    # Overlap q_ab = sum_k \sqrt{Q_ak * Q_bk} (Bhattacharyya coefficient) or simple dot
    overlaps = Q @ Q.T
    n = overlaps.shape[0]
    off_diag = overlaps[np.triu_indices(n, k=1)]
    return np.mean(off_diag)

def calculate_pca_explained(returns_window):
    try:
        pca = PCA(n_components=1)
        pca.fit(returns_window)
        return pca.explained_variance_ratio_[0]
    except:
        return 0

def run_analysis():
    close = download_data()
    if close is None: return
    
    returns = np.log(close / close.shift(1)).dropna()
    
    window_size = 60
    step = 5 
    
    solver = AdiabaticAttention(n_components=5, slow_decay=0.96)
    
    results = []
    
    print("Running Rolling Analysis with Non-Linear Attention...")
    for t in range(window_size, len(returns), step):
        date = returns.index[t]
        window = returns.iloc[t-window_size:t]
        
        # 1. Our Metric
        Q = solver.fit_step(window)
        parisi = calculate_parisi_order(Q)
        
        # 2. Benchmarks
        avg_corr = window.corr().values[np.triu_indices(window.shape[1], k=1)].mean()
        pc1_var = calculate_pca_explained(window)
        
        # 3. Targets: Volatility AND Skewness/Kurtosis (Tail Risk)
        if t + 20 < len(returns):
            future_rets = returns.iloc[t:t+20].mean(axis=1)
            future_vol = future_rets.std() * np.sqrt(252)
            # Negative Skewness is the hallmark of crashes
            future_skew = skew(future_rets)
            # Kurtosis
            future_kurt = kurtosis(future_rets)
        else:
            future_vol = np.nan
            future_skew = np.nan
            future_kurt = np.nan
            
        results.append({
            'Date': date,
            'Parisi_Order': parisi,
            'Avg_Corr': avg_corr,
            'PC1_Var': pc1_var,
            'Future_Vol_20d': future_vol,
            'Future_Skew_20d': future_skew,
            'Future_Kurt_20d': future_kurt
        })
        
    df = pd.DataFrame(results).set_index('Date').dropna()
    
    # --- Incremental Information Test ---
    print("\n" + "="*50)
    print("INCREMENTAL PREDICTIVE POWER TEST (Non-Linear)")
    print("="*50)
    
    for target in ['Future_Vol_20d', 'Future_Skew_20d', 'Future_Kurt_20d']:
        print(f"\nTarget: {target}")
        y = df[target]
        
        # Baseline
        X0 = sm.add_constant(df[['PC1_Var']])
        m0 = sm.OLS(y, X0).fit()
        
        # Full
        X1 = sm.add_constant(df[['PC1_Var', 'Parisi_Order']])
        m1 = sm.OLS(y, X1).fit()
        
        print(f"  Baseline R2: {m0.rsquared:.4f} -> Full R2: {m1.rsquared:.4f} (Diff: {m1.rsquared - m0.rsquared:.4f})")
        print(f"  Parisi t-stat: {m1.tvalues['Parisi_Order']:.2f}, p-value: {m1.pvalues['Parisi_Order']:.4f}")

    # --- Correlation Analysis ---
    corr_matrix = df[['Parisi_Order', 'Avg_Corr', 'PC1_Var', 'Future_Vol_20d']].corr()
    print("\nFeature Correlation Matrix:")
    print(corr_matrix)
    
    # --- Plotting ---
    print(corr_matrix)
    
    # --- Plotting ---
    if not os.path.exists('figures'): os.makedirs('figures')
    
    # Plot 1: The Divergence
    # We want to show where Parisi deviates from PCA
    
    # Normalize for plotting
    df_norm = (df - df.mean()) / df.std()
    
    plt.figure(figsize=(14, 6))
    plt.plot(df.index, df_norm['PC1_Var'], label='Standard PCA (Systemic Risk)', alpha=0.5, color='gray')
    plt.plot(df.index, df_norm['Parisi_Order'], label='Attention Topology (Ours)', color='red', linewidth=1.5)
    plt.title('Topology vs Linearity: Where do they diverge?')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Highlight periods of large divergence
    divergence = np.abs(df_norm['Parisi_Order'] - df_norm['PC1_Var'])
    # Find top divergence dates
    top_div = divergence.sort_values(ascending=False).head(5)
    print("\nTop Divergence Dates (Where our model differs most from PCA):")
    print(top_div)
    
    plt.savefig('figures/tier1_comparison_divergence.png')
    
    # Scatter Plot of Incremental Value
    plt.figure(figsize=(8, 8))
    # Color by Volatility
    plt.scatter(df['PC1_Var'], df['Parisi_Order'], c=df['Future_Vol_20d'], cmap='inferno', s=20, alpha=0.7)
    plt.xlabel('PCA First Component Variance (Standard)')
    plt.ylabel('Parisi Order Parameter (Attention)')
    plt.title('Orthogonality Check: Are we measuring the same thing?')
    plt.colorbar(label='Future Volatility')
    plt.grid(True, alpha=0.3)
    plt.savefig('figures/tier1_orthogonality.png')
    
    print("Analysis Complete. Figures saved to figures/")

if __name__ == "__main__":
    run_analysis()
