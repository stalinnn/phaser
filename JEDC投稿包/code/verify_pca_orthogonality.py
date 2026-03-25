import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA, NMF
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.special import softmax
import os

# --- 1. Reuse existing AdiabaticAttention class ---
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

# --- 2. Data Download Helper ---
def get_data():
    tickers = [
        "AAPL", "MSFT", "GOOGL", "AMZN", "META",
        "TSLA", "NVDA", "JPM", "V", "PG"
    ]
    start = "2015-01-01"
    end = "2024-01-01"
    
    print("Downloading US Tech/Bluechip Data for PCA Check...")
    try:
        data = yf.download(tickers, start=start, end=end, progress=False, auto_adjust=True)
        # Handle MultiIndex columns if yfinance returns them
        if isinstance(data.columns, pd.MultiIndex):
            close = data['Close']
        else:
            close = data['Close'] if 'Close' in data else data
            
        close = close.ffill().bfill()
        return close
    except Exception as e:
        print(f"Data download failed: {e}")
        return None

# --- 3. New Helper for Rolling PCA ---
def get_rolling_pca_var(returns_window):
    """Calculates the % variance explained by the 1st Principal Component"""
    try:
        pca = PCA(n_components=1)
        pca.fit(returns_window.fillna(0))
        return pca.explained_variance_ratio_[0]
    except:
        return np.nan

# --- 4. Main Verification Function ---
def run_pca_check():
    close = get_data()
    if close is None: return

    returns = np.log(close / close.shift(1)).dropna()
    
    solver = AdiabaticAttention()
    
    parisi_list = []
    pca_var_list = []
    dates = []
    
    window = 30
    print(f"Running PCA Comparison on {len(returns)} days...")
    
    for t in range(window, len(returns), 5): # Step=5 for speed
        w_ret = returns.iloc[t-window:t]
        
        # Calculate Parisi
        p = solver.get_parisi(w_ret)
        
        # Calculate PCA
        pca_v = get_rolling_pca_var(w_ret)
        
        if not np.isnan(p) and not np.isnan(pca_v):
            parisi_list.append(p)
            pca_var_list.append(pca_v)
            dates.append(returns.index[t])
            
    df_compare = pd.DataFrame({
        'Parisi': parisi_list,
        'PCA_Var_Ratio': pca_var_list
    }, index=dates)
    
    # Calc Correlation
    corr = df_compare.corr().iloc[0,1]
    print(f"\nCorrelation(Parisi, PCA_PC1) = {corr:.4f}")
    
    # Save Results
    if not os.path.exists('JEDC_Submission_Package/results'):
        os.makedirs('JEDC_Submission_Package/results')
        
    with open('JEDC_Submission_Package/results/pca_check_result.txt', 'w') as f:
        f.write(f"Parisi vs PCA Orthogonality Check\n")
        f.write(f"Correlation: {corr:.4f}\n")
        f.write(f"Interpretation: Low correlation (<0.5) confirms Parisi captures non-linear structure distinct from PCA.")

    # Plot
    if not os.path.exists('JEDC_Submission_Package/figures'):
        os.makedirs('JEDC_Submission_Package/figures')
        
    plt.figure(figsize=(10, 6))
    plt.scatter(df_compare['PCA_Var_Ratio'], df_compare['Parisi'], alpha=0.3, color='darkgreen')
    plt.title(f'Orthogonality Check: Parisi vs PCA (Linear Risk)\nCorrelation r = {corr:.2f}')
    plt.xlabel('PCA Explained Variance (Linear Systemic Risk)')
    plt.ylabel('Parisi Order Parameter (Non-linear Crowding)')
    plt.grid(True, alpha=0.3)
    plt.savefig('JEDC_Submission_Package/figures/pca_orthogonality_check.png')
    print("Saved plot to JEDC_Submission_Package/figures/pca_orthogonality_check.png")

if __name__ == "__main__":
    run_pca_check()
