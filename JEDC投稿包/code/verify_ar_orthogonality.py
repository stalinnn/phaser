import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.decomposition import NMF, PCA
from scipy.special import softmax
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr, spearmanr

# --------------------------
# 1. Reuse Solver Classes
# --------------------------
class MetricsSolver:
    def __init__(self, n_components=4):
        self.n_components = n_components
        
    def get_tfi(self, returns_window):
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

    def get_ar(self, returns_window):
        if returns_window.shape[1] < self.n_components: return 0
        try:
            X = (returns_window - returns_window.mean()) / returns_window.std()
            X = X.dropna(axis=1)
            pca = PCA()
            pca.fit(X)
            explained = pca.explained_variance_ratio_
            ar = np.sum(explained[:self.n_components])
            return ar
        except:
            return 0

# --------------------------
# 2. Main Logic
# --------------------------
def main():
    print("Fetching data...")
    tickers = ["MSFT", "AAPL", "AMZN", "GOOG", "JPM", "XOM", "BAC", "WMT", "INTC", "CSCO", "C", "PFE", "KO"]
    try:
        data = yf.download(tickers, start="2005-01-01", end="2024-01-01", progress=False, auto_adjust=True)['Close']
        returns = np.log(data / data.shift(1)).dropna()
    except:
        print("Data fetch failed.")
        return

    print("Calculating Metrics...")
    solver = MetricsSolver(n_components=3)
    
    tfi_list = []
    ar_list = []
    dates = []
    
    window = 60
    for t in range(window, len(returns), 5): # Step 5
        w = returns.iloc[t-window:t]
        dates.append(returns.index[t])
        tfi_list.append(solver.get_tfi(w))
        ar_list.append(solver.get_ar(w))
        
    df = pd.DataFrame({'TFI': tfi_list, 'AR': ar_list}, index=dates)
    
    # --------------------------
    # 3. Orthogonality Stats
    # --------------------------
    p_corr, _ = pearsonr(df['TFI'], df['AR'])
    s_corr, _ = spearmanr(df['TFI'], df['AR'])
    
    print(f"\n--- Orthogonality Check ---")
    print(f"Pearson Correlation (Linear):   {p_corr:.4f}")
    print(f"Spearman Correlation (Rank):    {s_corr:.4f}")
    
    # R-squared
    print(f"Shared Variance (R^2):          {p_corr**2:.4f}")
    print(f"Unique Variance (1 - R^2):      {1 - p_corr**2:.4f}")
    
    # --------------------------
    # 4. Visualization
    # --------------------------
    plt.figure(figsize=(10, 6))
    sns.regplot(x='AR', y='TFI', data=df, scatter_kws={'alpha':0.3, 's':10}, line_kws={'color':'red'})
    plt.title(f"TFI vs AR Scatter Plot (r={p_corr:.2f})")
    plt.xlabel("Absorption Ratio (Linear PCA)")
    plt.ylabel("Topological Fragility Index (Non-linear Softmax)")
    plt.grid(True, alpha=0.3)
    
    output_path = "JEDC_Submission_Package/figures/ar_orthogonality_check.png"
    plt.savefig(output_path)
    print(f"\nSaved scatter plot to {output_path}")

if __name__ == "__main__":
    main()
