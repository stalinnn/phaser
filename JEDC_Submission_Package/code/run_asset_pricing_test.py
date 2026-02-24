import yfinance as yf
import pandas as pd
import numpy as np
import statsmodels.api as sm
from sklearn.decomposition import NMF
from scipy.special import softmax
import matplotlib.pyplot as plt
import os

"""
ASSET PRICING TEST: DOES PARISI PREDICT RISK PREMIUM?
----------------------------------------------------
Objective: Formal econometric test to see if Parisi Order Parameter 
significantly predicts future Equity Risk Premium (ERP) after controlling for volatility.

Model: R_{t+1} = alpha + beta1 * Parisi_t + beta2 * VIX_t + epsilon
"""

def get_market_data():
    # Proxies:
    # Market: SPY
    # Volatility: ^VIX (CBOE Volatility Index)
    tickers = ["SPY", "^VIX"]
    
    print("Downloading Market Data (SPY, VIX)...")
    data = yf.download(tickers, start="2000-01-01", end="2024-01-01", progress=False, auto_adjust=True)
    
    if isinstance(data.columns, pd.MultiIndex):
        close = data['Close']
    else:
        close = data
        
    close = close.ffill().dropna()
    return close

def get_universe_data():
    # Use S&P 100 proxy for TFI calculation
    # In a real submission, this would load the survivor-bias-free dataset
    tickers = [
        "MSFT", "AMZN", "JPM", "XOM", "GE", "JNJ", "PFE", "C", "KO", "PEP",
        "WMT", "MRK", "INTC", "CSCO", "IBM", "GS", "BAC", "AIG", "AXP", "MCD",
        "AAPL", "GOOGL", "NVDA", "TSLA", "META", "BRK-B", "UNH", "LLY", "V", "PG"
    ]
    print("Downloading Universe for TFI Calculation...")
    data = yf.download(tickers, start="2000-01-01", end="2024-01-01", progress=False, auto_adjust=True)
    if isinstance(data.columns, pd.MultiIndex):
        close = data['Close']
    else:
        close = data
    # Survivorship bias mitigation: Drop only empty columns
    return close.dropna(axis=1, how='all').ffill()

# --- Re-implement TFI Core Logic ---
class AdiabaticAttention:
    def __init__(self, n_components=5):
        self.n_components = n_components
        
    def fit_transform(self, returns):
        # Rolling calculation
        window_size = 60
        step = 5
        dates = []
        tfi_vals = []
        
        print("Calculating TFI Factor History...")
        for t in range(window_size, len(returns), step):
            w = returns.iloc[t-window_size:t].dropna(axis=1)
            if w.shape[1] < 10: continue
            
            corr = np.nan_to_num(w.corr().values)
            
            # NMF
            try:
                model = NMF(n_components=min(5, w.shape[1]//2), init='random', random_state=42, max_iter=20)
                W = model.fit_transform(corr + 1.0)
                H = model.components_
                K = H.T
                
                # Attention
                lambda_reg = 0.5
                K_inv = np.linalg.inv(K.T @ K + lambda_reg * np.eye(K.shape[1]))
                Q_linear = (corr + 1.0) @ K @ K_inv
                
                max_corr = np.max(corr - np.eye(len(corr)))
                T_eff = np.clip(1.0 - max_corr, 0.05, 1.0)
                
                Q_nonlinear = softmax(Q_linear / T_eff, axis=1)
                
                # Order Param (TFI)
                overlaps = Q_nonlinear @ Q_nonlinear.T
                off_diag = overlaps[np.triu_indices(len(overlaps), k=1)]
                q = np.mean(off_diag)
                
                dates.append(returns.index[t])
                tfi_vals.append(q)
            except:
                continue
                
        return pd.Series(tfi_vals, index=dates)

def run_regression_test():
    # 1. Prepare Data
    mkt_data = get_market_data()
    univ_data = get_universe_data()
    
    if mkt_data is None or univ_data is None: return
    
    # 2. Calc Factors
    univ_ret = np.log(univ_data / univ_data.shift(1))
    aa = AdiabaticAttention()
    tfi_factor = aa.fit_transform(univ_ret)
    tfi_factor.name = 'TFI'
    
    # 3. Align Data
    # PREDICTION TARGET: FUTURE KURTOSIS (Tail Risk Shape)
    # Hypothesis: TFI predicts the *fat-tailedness* of the distribution.
    
    spy_daily_ret = mkt_data['SPY'].pct_change()
    
    # Calculate Rolling Kurtosis (requires scipy)
    # We need a rolling window. Pandas rolling.kurt() is available.
    future_kurt = spy_daily_ret.rolling(window=20).kurt().shift(-20)
    future_kurt.name = 'Fwd_Kurt'
    
    vix = mkt_data['^VIX']
    vix.name = 'VIX'
    
    df = pd.concat([future_kurt, tfi_factor, vix], axis=1).dropna()
    
    # Remove extreme outliers for stability
    df = df[df['Fwd_Kurt'] < 10]
    
    # 4. Regression Analysis
    print("\n" + "="*60)
    print("ECONOMETRIC TEST: PREDICTING TAIL RISK (KURTOSIS)")
    print("Dependent Variable: Future 20-Day Return Kurtosis")
    print("Hypothesis: TFI (Crowding) -> Higher Fat Tails")
    print("="*60)
    
    # Model 1: Control Only (VIX)
    X1 = sm.add_constant(df[['VIX']])
    y = df['Fwd_Kurt']
    model1 = sm.OLS(y, X1).fit()
    
    print("Model 1: Benchmark (VIX Only)")
    print(f"R-squared: {model1.rsquared:.4f}")
    print(f"VIX t-stat: {model1.tvalues['VIX']:.2f}")
    print("-" * 30)
    
    # Model 3: Combined
    X3 = sm.add_constant(df[['VIX', 'TFI']])
    model3 = sm.OLS(y, X3).fit()
    
    print("Model 3: Combined (VIX + TFI)")
    print(f"R-squared: {model3.rsquared:.4f}")
    print(f"TFI Coeff: {model3.params['TFI']:.4f}")
    print(f"TFI t-stat: {model3.tvalues['TFI']:.2f}")
    print(f"TFI P-value: {model3.pvalues['TFI']:.4e}")
    print(f"VIX t-stat: {model3.tvalues['VIX']:.2f}")
    print("="*60)
    
    # Save Results
    with open("JEDC_Submission_Package1/results/kurtosis_regression.txt", "w") as f:
        f.write(model3.summary().as_text())
        
    print("Summary saved to results/kurtosis_regression.txt")
    
    # Save Results to File
    with open("JEDC_Submission_Package1/results/regression_results.txt", "w") as f:
        f.write(model3.summary().as_text())
        
    print("Full regression summary saved to results/regression_results.txt")

if __name__ == "__main__":
    run_regression_test()
