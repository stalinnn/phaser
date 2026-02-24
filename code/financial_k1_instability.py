
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import matplotlib.dates as mdates

# Set style
plt.style.use('seaborn-v0_8-paper')
sns.set_context("paper", font_scale=1.5)

def get_tickers():
    # Using the robust list from empirical_finance_long.py
    sectors = ["XLE", "XLF", "XLU", "XLI", "XLK", "XLV", "XLY", "XLP", "XLB"]
    blue_chips = [
        "GE", "IBM", "XOM", "PG", "KO", "JNJ", "PFE", "MRK", "JPM", "BAC", 
        "C", "WFC", "AIG", "AXP", "MSFT", "INTC", "CSCO", "ORCL", "QCOM", "ADBE",
        "WMT", "HD", "MCD", "DIS", "NKE", "PEP", "MMM", "BA", "CAT", "RTX",
        "CVX", "COP", "SLB", "HAL", "BMY", "LLY", "ABT", "UNH", "AEP", "D",
        "SO", "DUK", "EXC", "FDX", "UPS", "CL", "KMB", "MO", "PM", "T",
        "VZ", "CMCSA", "F", "GM", "HON", "EMR", "ITW", "DE", "GD", "LMT",
        "NOC", "GS", "MS", "USB", "BK", "SCHW", "AMT", "SPG", "PLD", "PSA",
        "ADP", "PAYX"
    ]
    return list(set(sectors + blue_chips))

def fetch_data():
    tickers = get_tickers()
    print(f"Fetching data for {len(tickers)} assets...")
    try:
        # Download data
        data = yf.download(tickers, start="2000-01-01", end="2023-01-01", progress=False)['Close']
        
        # Clean data
        # Remove columns with too many NaNs
        missing = data.isna().mean()
        data = data[missing[missing < 0.2].index]
        data = data.ffill().bfill()
        
        # Log returns
        returns = np.log(data / data.shift(1)).dropna()
        
        # Also fetch VIX for reference
        vix = yf.download("^VIX", start="2000-01-01", end="2023-01-01", progress=False)['Close']
        if isinstance(vix, pd.DataFrame):
            vix = vix.iloc[:, 0]
            
        return returns, vix
    except Exception as e:
        print(f"Error fetching data: {e}")
        return None, None

def compute_metrics(returns, window=40):
    """
    Compute:
    1. Geometric Entropy (S) - The existing metric
    2. Median Serial Correlation (rho) - The PROPOSED K=1 signature
    3. Market Volatility (sigma)
    """
    print("Computing metrics...")
    dates = returns.index[window:]
    entropy = []
    serial_corr = []
    volatility = []
    
    # Pre-calculate rolling windows to speed up
    # Rolling correlation is computationally expensive if done per asset in a loop
    # We will do a simplified approach
    
    for i in range(window, len(returns)):
        if i % 500 == 0:
            print(f"Processing step {i}/{len(returns)}")
            
        # Window slice
        window_returns = returns.iloc[i-window:i]
        
        # 1. Geometric Entropy
        corr_matrix = window_returns.corr()
        # Handle NaNs in correlation matrix (const prices)
        corr_matrix = corr_matrix.fillna(0)
        
        try:
            eigenvalues = np.linalg.eigvalsh(corr_matrix)
            eigenvalues = eigenvalues[eigenvalues > 1e-10] # Filter numerical noise
            prob = eigenvalues / eigenvalues.sum()
            ent = -np.sum(prob * np.log(prob))
            entropy.append(ent)
        except:
            entropy.append(np.nan)
            
        # 2. Serial Correlation (The K=1 Signature)
        # We calculate the lag-1 autocorrelation for EACH asset, then take the median
        # If the market is in "Overshoot" mode, this should dip negative
        
        # Vectorized autocorrelation for the window
        # corr(x_t, x_{t-1})
        current = window_returns.iloc[1:].values
        lagged = window_returns.iloc[:-1].values
        
        # Compute correlation per asset
        # Center the data
        curr_mean = current.mean(axis=0)
        lag_mean = lagged.mean(axis=0)
        
        numerator = np.sum((current - curr_mean) * (lagged - lag_mean), axis=0)
        denominator = np.sqrt(np.sum((current - curr_mean)**2, axis=0) * np.sum((lagged - lag_mean)**2, axis=0))
        
        # Avoid div by zero
        autocorrs = np.divide(numerator, denominator, out=np.zeros_like(numerator), where=denominator!=0)
        
        # We take the MEDIAN to be robust to outliers
        serial_corr.append(np.median(autocorrs))
        
        # 3. Volatility (Mean standard deviation across assets)
        vol = window_returns.std().median()
        volatility.append(vol)
        
    results = pd.DataFrame({
        'Entropy': entropy,
        'SerialCorr': serial_corr,
        'Volatility': volatility
    }, index=dates)
    
    # Smoothing
    results['Entropy_Smooth'] = results['Entropy'].rolling(10).mean()
    results['SerialCorr_Smooth'] = results['SerialCorr'].rolling(10).mean()
    
    return results

def plot_k1_analysis(results, vix, save_path):
    print("Plotting analysis...")
    
    # Align VIX
    common_idx = results.index.intersection(vix.index)
    results = results.loc[common_idx]
    vix = vix.loc[common_idx]
    
    fig = plt.figure(figsize=(15, 12))
    gs = fig.add_gridspec(3, 2)
    
    # 1. The Timeline (2007-2009 Focus)
    ax1 = fig.add_subplot(gs[0, :])
    
    # Focus on 2008 Crisis
    start_date = '2008-01-01'
    end_date = '2009-06-01'
    subset = results.loc[start_date:end_date]
    subset_vix = vix.loc[start_date:end_date]
    
    # Plot Entropy
    ax1.plot(subset.index, subset['Entropy_Smooth'], color='#2ecc71', label='Geometric Entropy (S)', linewidth=2)
    ax1.set_ylabel('Entropy (Structure)', color='#2ecc71', fontweight='bold')
    ax1.tick_params(axis='y', labelcolor='#2ecc71')
    
    # Twin axis for Serial Correlation
    ax2 = ax1.twinx()
    ax2.plot(subset.index, subset['SerialCorr_Smooth'], color='#e74c3c', label='Serial Correlation (rho)', linewidth=1.5, linestyle='--')
    
    # Highlight the "Danger Zone" (Negative Correlation)
    ax2.axhline(0, color='gray', linestyle=':', alpha=0.5)
    ax2.fill_between(subset.index, subset['SerialCorr_Smooth'], 0, where=(subset['SerialCorr_Smooth'] < 0), color='#e74c3c', alpha=0.2, label='K=1 Instability (Overshoot)')
    
    ax2.set_ylabel('Serial Correlation (K=1 Proxy)', color='#e74c3c', fontweight='bold')
    ax2.tick_params(axis='y', labelcolor='#e74c3c')
    
    ax1.set_title('2008 Crisis Micro-Structure: The "Blind Coordination" Precursor', fontsize=14, fontweight='bold')
    
    # Mark Lehman
    lehman_date = datetime.strptime('2008-09-15', '%Y-%m-%d')
    if lehman_date in subset.index:
        ax1.axvline(lehman_date, color='black', linestyle='-', alpha=0.8)
        ax1.text(lehman_date, subset['Entropy_Smooth'].max(), ' Lehman Collapse', verticalalignment='top')

    # 2. Phase Portrait: Entropy vs Serial Correlation
    ax3 = fig.add_subplot(gs[1, 0])
    
    # Color by Volatility
    sc = ax3.scatter(results['Entropy_Smooth'], results['SerialCorr_Smooth'], 
                     c=np.log(results['Volatility']), cmap='magma_r', alpha=0.5, s=10)
    
    ax3.set_xlabel('Geometric Entropy (S)')
    ax3.set_ylabel('Serial Correlation (rho)')
    ax3.set_title('Phase Space: Stability vs Instability')
    ax3.axhline(0, color='black', linestyle='--')
    
    # Annotate Quadrants
    ax3.text(results['Entropy_Smooth'].min(), 0.1, 'Random Walk\n(Normal)', ha='left', va='bottom', fontsize=10)
    ax3.text(results['Entropy_Smooth'].min(), -0.15, 'K=1 Instability\n(Blind Overshoot)', ha='left', va='top', color='red', fontsize=10, fontweight='bold')
    
    plt.colorbar(sc, ax=ax3, label='Log Volatility')

    # 3. 2020 Covid Crisis Zoom
    ax4 = fig.add_subplot(gs[1, 1])
    start_2020 = '2020-01-01'
    end_2020 = '2020-06-01'
    sub2020 = results.loc[start_2020:end_2020]
    
    ax4.plot(sub2020.index, sub2020['Entropy_Smooth'], color='#2ecc71', label='Entropy')
    ax4_twin = ax4.twinx()
    ax4_twin.plot(sub2020.index, sub2020['SerialCorr_Smooth'], color='#e74c3c', linestyle='--', label='Serial Corr')
    ax4_twin.fill_between(sub2020.index, sub2020['SerialCorr_Smooth'], 0, where=(sub2020['SerialCorr_Smooth'] < 0), color='#e74c3c', alpha=0.2)
    
    ax4.set_title('2020 Covid Melt-down: Instantaneous Collapse')
    ax4.set_ylabel('Entropy', color='#2ecc71')
    ax4_twin.set_ylabel('Serial Corr', color='#e74c3c')
    
    # 4. Lead-Lag Analysis (Cross Correlation)
    ax5 = fig.add_subplot(gs[2, :])
    
    # Compute Cross Correlation between Entropy and SerialCorr
    # We want to see if SerialCorr dips BEFORE Entropy drops?
    # Actually, let's plot VIX vs Serial Corr
    
    ax5.scatter(vix, results['SerialCorr_Smooth'], alpha=0.1, color='purple')
    ax5.set_xlabel('VIX (Volatility Index)')
    ax5.set_ylabel('Serial Correlation')
    ax5.set_title('The Cost of Panic: Volatility induces Negative Correlation (Overshoot)')
    ax5.axhline(0, color='black', linestyle='--')
    
    # Add trend line
    z = np.polyfit(vix, results['SerialCorr_Smooth'].fillna(0), 1)
    p = np.poly1d(z)
    ax5.plot(vix, p(vix), "k--", alpha=0.8)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    print(f"Plot saved to {save_path}")

if __name__ == "__main__":
    returns, vix = fetch_data()
    if returns is not None:
        results = compute_metrics(returns)
        plot_k1_analysis(results, vix, "figures/k1_instability_proof.png")

