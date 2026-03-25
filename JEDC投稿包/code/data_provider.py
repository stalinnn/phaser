import pandas as pd
import yfinance as yf
import numpy as np
import os
import warnings

warnings.filterwarnings('ignore')

"""
JEDC DATA PROVIDER MODULE
-------------------------
Purpose: Centralize data loading to address survivorship bias.
Strategy:
1. Check for local high-quality database (CRSP/Bloomberg format) first.
2. Fallback to yfinance with an EXPANDED universe (Historical + Current).
3. Handle delisted tickers gracefully.
"""

# -----------------------------------------------------------------------------
# EXPANDED UNIVERSE: Current S&P 100 + Major Historical Constituents (2000-2024)
# -----------------------------------------------------------------------------
# Note: yfinance may not fetch data for fully liquidated companies (e.g., Lehman),
# but including them allows this script to work immediately if mapped to a 
# survivorship-bias-free local database file.
# -----------------------------------------------------------------------------
HISTORICAL_UNIVERSE = [
    # --- Current Giants ---
    "MSFT", "AMZN", "JPM", "XOM", "GE", "JNJ", "PFE", "C", "KO", "PEP",
    "WMT", "MRK", "INTC", "CSCO", "IBM", "GS", "BAC", "AIG", "AXP", "MCD",
    "AAPL", "GOOGL", "NVDA", "TSLA", "META", "BRK-B", "UNH", "LLY", "V", "PG",
    "HD", "MA", "CVX", "ABBV", "ADBE", "NFLX", "DIS", "CMCSA", "TXN", "PM", 
    "HON", "QCOM", "AMGN", "CAT", "SPGI", "MS", "BA", "MMM", "T", "VZ",
    
    # --- Major Historical / Acquired / Merged / Rebranded (2000-2024) ---
    # Tech/Telecom
    "ORCL", "CRM", "AMD", "QCOM", "AVGO", "TXN", "MU", "ADI", "LRCX",
    "YHOO", "AOL", "S", "LU", "MOT", "NOK", "BB", "HPQ", "DELL", "EMC",
    
    # Financials (Crisis Era Focus)
    "LEH", "BSC", "WM", "FNM", "FRE", "MER", "WB", "NCC", "WFC", "USB",
    "PNC", "BK", "STT", "BLK", "COF", "MET", "PRU", "TRV", "ALL",
    
    # Industrials & Energy
    "RTX", "LMT", "GD", "NOC", "BA", "GE", "MMM", "HON", "CAT", "DE",
    "COP", "SLB", "HAL", "BKR", "EOG", "OXY", "KMI", "WMB", "PSX", "MPC",
    
    # Consumer & Pharma
    "MO", "PM", "BTI", "CL", "KMB", "EL", "NKE", "SBUX", "LOW", "TGT",
    "COST", "WBA", "CVS", "CI", "HUM", "CNC", "BMY", "GILD", "REGN", "VRTX"
]

def load_market_data(
    start_date="2000-01-01", 
    end_date="2024-01-01", 
    source_file="data/market_data_cleaned.csv",
    force_download=False
):
    """
    Robust data loader for JEDC submission.
    
    Priority:
    1. Local CSV (source_file) - Expected format: Index=Date, Columns=Tickers, Values=Adj Close
    2. yfinance download (Fallback)
    """
    
    # 1. Try Local File (Best for Reproducibility with Proprietary Data)
    if not force_download and os.path.exists(source_file):
        print(f"[Data] Loading local file: {source_file}")
        try:
            df = pd.read_csv(source_file, index_col=0, parse_dates=True)
            df = df.sort_index()
            # Filter Date Range
            mask = (df.index >= start_date) & (df.index <= end_date)
            df = df.loc[mask]
            
            # Basic cleaning
            df = df.dropna(axis=1, how='all') # Drop empty tickers
            print(f"[Data] Loaded {df.shape[1]} assets from {df.index[0].date()} to {df.index[-1].date()}")
            return df
        except Exception as e:
            print(f"[Data] Error reading local file: {e}. Falling back to download.")

    # 2. Fallback to yfinance
    print(f"[Data] Downloading from Yahoo Finance (Note: Contains Survivorship Bias)...")
    
    # Clean ticker list
    tickers = list(set(HISTORICAL_UNIVERSE))
    
    try:
        # Download in one go (yfinance handles batching internally now)
        data = yf.download(
            tickers, 
            start=start_date, 
            end=end_date, 
            progress=False, 
            auto_adjust=True,
            threads=True
        )
        
        # Handle MultiIndex if present
        if isinstance(data.columns, pd.MultiIndex):
            # Prefer 'Close' or 'Adj Close' (auto_adjust=True makes Close=Adj Close)
            if 'Close' in data.columns.levels[0]:
                close = data['Close']
            elif 'Adj Close' in data.columns.levels[0]:
                close = data['Adj Close']
            else:
                close = data.iloc[:, :len(tickers)] # Fallback
        else:
            close = data

        # 3. Clean Data for Matrix Operations
        # Rule: We do NOT drop columns with NaNs, as that kills historical stocks.
        # We only drop columns that are effectively empty.
        close = close.dropna(axis=1, how='all')
        
        # Forward fill to handle trading halts / holidays
        close = close.ffill()
        
        # Note on NaNs:
        # In the main model, we must handle NaNs dynamically (per window), 
        # NOT drop them globally here.
        
        print(f"[Data] Download complete. Shape: {close.shape}")
        return close
        
    except Exception as e:
        print(f"[Error] Data download failed: {e}")
        return None

def get_sp500_benchmark(start_date, end_date):
    """Get S&P 500 benchmark for comparison"""
    try:
        spy = yf.download("SPY", start=start_date, end=end_date, progress=False, auto_adjust=True)
        if isinstance(spy.columns, pd.MultiIndex):
            return spy['Close']
        return spy
    except:
        return None

if __name__ == "__main__":
    # Test run
    df = load_market_data(start_date="2020-01-01", end_date="2021-01-01")
    if df is not None:
        print("Sample Data:\n", df.iloc[:5, :5])
