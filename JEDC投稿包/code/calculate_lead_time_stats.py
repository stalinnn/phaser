import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.decomposition import NMF, PCA
from scipy.special import softmax

def get_data(market='US'):
    if market == 'US':
        tickers = ["MSFT", "AMZN", "JPM", "XOM", "GE", "JNJ", "PFE", "C", "KO", "PEP"]
        start = "2000-01-01"
    else:
        # A-share proxies on Yahoo Finance
        tickers = ["600519.SS", "601318.SS", "600036.SS", "601166.SS", "600900.SS"]
        start = "2010-01-01"
        
    try:
        data = yf.download(tickers, start=start, end="2024-06-01", progress=False, auto_adjust=True)
        if isinstance(data.columns, pd.MultiIndex):
            close = data['Close']
        else:
            close = data
        # Basic cleaning
        close = close.dropna(axis=1, thresh=int(len(close)*0.8)).ffill().bfill()
        return close
    except:
        return None

class MetricsSolver:
    def __init__(self, n_components=4):
        self.n_components = n_components
        
    def get_tfi(self, returns_window):
        # TFI (Parisi) Calculation
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
        # Absorption Ratio (AR) Calculation
        # AR = Variance explained by top K eigenvectors / Total Variance
        if returns_window.shape[1] < self.n_components: return 0
        try:
            # Standardize for PCA
            X = (returns_window - returns_window.mean()) / returns_window.std()
            X = X.dropna(axis=1)
            pca = PCA()
            pca.fit(X)
            explained = pca.explained_variance_ratio_
            # Sum top K (e.g. top 20% or fixed number)
            # Kritzman uses ~1/5 of assets. Here we use n_components (4) for consistency.
            ar = np.sum(explained[:self.n_components])
            return ar
        except:
            return 0

def calc_lead_stats(market_name):
    print(f"\n==========================================")
    print(f"  Analyzing {market_name} Market (Lead Time Test)")
    print(f"==========================================")
    
    close = get_data(market_name)
    if close is None: return
    
    # 1. Market Index & Returns
    market_index = close.mean(axis=1)
    returns = np.log(close / close.shift(1)).dropna()
    
    solver = MetricsSolver(n_components=max(2, int(close.shape[1]/3))) # Adaptive K
    
    tfi_list = []
    ar_list = []
    dates = []
    
    window = 40 # 40-day window
    
    # Rolling calculation (step 5 for speed)
    for t in range(window, len(returns), 5):
        w = returns.iloc[t-window:t]
        dates.append(returns.index[t])
        tfi_list.append(solver.get_tfi(w))
        ar_list.append(solver.get_ar(w))
        
    df = pd.DataFrame({'TFI': tfi_list, 'AR': ar_list}, index=dates)
    df = df.reindex(market_index.index).ffill()
    
    # Add Volatility (HV)
    df['Vol'] = market_index.pct_change().rolling(30).std() * np.sqrt(252)
    
    # 2. Define Crashes (Drawdown > 15%)
    rolling_max = market_index.rolling(250, min_periods=1).max()
    drawdown = (market_index - rolling_max) / rolling_max
    
    # Crash Start Identification
    is_crash = (drawdown < -0.15)
    crash_starts = []
    last_crash_date = pd.Timestamp('1900-01-01')
    
    for date in is_crash[is_crash].index:
        if (date - last_crash_date).days > 365: # Distinct crashes
            crash_starts.append(date)
            last_crash_date = date
            
    # 3. Calculate Lead Times
    results = []
    
    # Dynamic Thresholds (Rolling 1-year quantile)
    # This simulates a real-time adaptive strategy
    df['TFI_Thresh'] = df['TFI'].rolling(252, min_periods=60).quantile(0.90)
    df['AR_Thresh']  = df['AR'].rolling(252, min_periods=60).quantile(0.90)
    
    for crash_date in crash_starts:
        # Observation Window: 60 days before crash
        obs_start = crash_date - pd.Timedelta(days=60)
        w = df.loc[obs_start:crash_date]
        if len(w) == 0: continue
        
        # Check Signals
        # Signal = Indicator > Adaptive Threshold
        tfi_sig = w[w['TFI'] > w['TFI_Thresh']].index
        ar_sig  = w[w['AR']  > w['AR_Thresh']].index
        
        tfi_lead = (crash_date - tfi_sig[0]).days if len(tfi_sig) > 0 else 0
        ar_lead  = (crash_date - ar_sig[0]).days  if len(ar_sig)  > 0 else 0
        
        # Only log if at least one detected it (to filter super-fast crashes)
        if tfi_lead > 0 or ar_lead > 0:
            results.append({
                'Crash': crash_date.strftime('%Y-%m-%d'),
                'TFI Lead': tfi_lead,
                'AR Lead': ar_lead
            })
            
    res_df = pd.DataFrame(results)
    print(res_df)
    
    if not res_df.empty:
        avg_tfi = res_df['TFI Lead'].mean()
        avg_ar = res_df['AR Lead'].mean()
        print("-" * 40)
        print(f"Avg TFI Lead Time: {avg_tfi:.1f} days")
        print(f"Avg AR  Lead Time: {avg_ar:.1f} days")
        print(f"TFI Advantage: +{avg_tfi - avg_ar:.1f} days")
        print("-" * 40)

if __name__ == "__main__":
    calc_lead_stats("US")
    calc_lead_stats("China")
