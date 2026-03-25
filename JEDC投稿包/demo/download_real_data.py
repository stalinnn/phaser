import yfinance as yf
import pandas as pd
import os
import time

# --- CONFIG ---
START_DATE = "2000-01-01"
END_DATE = pd.Timestamp.now().strftime("%Y-%m-%d")
DATA_DIR = "demo/data"

if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)

# --- TICKERS ---
# US S&P 100 Proxy
US_TICKERS = [
    "MSFT", "AMZN", "JPM", "XOM", "GE", "JNJ", "PFE", "C", "KO", "PEP",
    "WMT", "MRK", "INTC", "CSCO", "IBM", "GS", "BAC", "AIG", "AXP", "MCD",
    "AAPL", "GOOGL", "NVDA", "TSLA", "META", "BRK-B", "UNH", "LLY", "V", "PG",
    "HD", "MA", "CVX", "ABBV", "ADBE", "NFLX", "DIS", "CMCSA", "TXN", "PM", 
    "HON", "QCOM", "AMGN", "CAT", "SPGI", "MS", "BA", "MMM", "T", "VZ"
]

# China A50 Proxies
# Yahoo symbols for A-shares: 600519.SS, 000858.SZ
CN_TICKERS = [
    "600519.SS", "601318.SS", "600036.SS", "600276.SS", "601012.SS",
    "600900.SS", "600887.SS", "600030.SS", "603288.SS", "601888.SS",
    "000858.SZ", "000333.SZ", "002415.SZ", "002594.SZ", "000651.SZ",
    "000002.SZ", "000725.SZ", "000001.SZ", "601166.SS", "601328.SS"
]

# Crypto
CRYPTO_TICKERS = [
    "BTC-USD", "ETH-USD", "BNB-USD", "SOL-USD", "XRP-USD", "ADA-USD",
    "DOGE-USD", "AVAX-USD", "TRX-USD", "DOT-USD", "MATIC-USD", "LTC-USD"
]

def download_and_save(tickers, name):
    print(f"\n--- Downloading {name} ({len(tickers)} assets) ---")
    print(f"Time range: {START_DATE} to {END_DATE}")
    
    try:
        # Tqdm is optional but nice for yf
        data = yf.download(
            tickers, 
            start=START_DATE, 
            end=END_DATE, 
            group_by='ticker', 
            auto_adjust=True,
            threads=True, # Multi-threading enabled
            progress=True
        )
        
        # YFinance returns a MultiIndex (Ticker, OHLCV) if multiple tickers
        # We need to extract just 'Close'
        if isinstance(data.columns, pd.MultiIndex):
            # Try 'Close' first, then 'Adj Close' (auto_adjust=True usually gives just Close/Open/...)
            try:
                # If group_by='ticker', columns are (Ticker, PriceType)
                # But yf.download structure varies by version. 
                # Let's try standard structure: data['Close'] if group_by='column' (default)
                # We used group_by='ticker', so top level is Ticker.
                # Actually, group_by='column' is easier to extract Close.
                pass
            except:
                pass
                
        # Re-download with group_by='column' for easier extraction
        data = yf.download(
            tickers, 
            start=START_DATE, 
            end=END_DATE, 
            auto_adjust=True,
            threads=True
        )
        
        if isinstance(data.columns, pd.MultiIndex):
            close = data['Close']
        else:
            close = data
            
        # Clean: Drop columns with all NaNs
        close = close.dropna(axis=1, how='all')
        
        # Fill missing: ffill then bfill
        close = close.ffill().bfill()
        
        # Save
        filename = f"{name}_fallback.csv"
        path = os.path.join(DATA_DIR, filename)
        close.to_csv(path)
        print(f"✅ Saved to {path} (Shape: {close.shape})")
        
        # Also save to warehouse directly to skip "fallback loading" step in App
        warehouse_path = os.path.join(DATA_DIR, "warehouse", f"{name}.csv") # Note: Name mapping might differ in App
        # App mapping: 
        # CN -> CN_A50.csv
        # US -> US_SP100.csv
        # CRYPTO -> CRYPTO.csv
        
        if name == "A_share": w_name = "CN_A50.csv"
        elif name == "US_SP100": w_name = "US_SP100.csv"
        else: w_name = "CRYPTO.csv"
        
        w_path = os.path.join(DATA_DIR, "warehouse", w_name)
        if not os.path.exists(os.path.dirname(w_path)):
            os.makedirs(os.path.dirname(w_path))
            
        close.to_csv(w_path)
        print(f"✅ Also synced to Warehouse: {w_path}")
        
    except Exception as e:
        print(f"❌ Failed to download {name}: {e}")

if __name__ == "__main__":
    # Ensure warehouse dir exists
    if not os.path.exists("demo/data/warehouse"):
        os.makedirs("demo/data/warehouse")
        
    download_and_save(US_TICKERS, "US_SP100")
    download_and_save(CN_TICKERS, "A_share")
    download_and_save(CRYPTO_TICKERS, "CRYPTO")
    
    print("\nAll downloads complete. You can now run 'streamlit run demo/app.py'")
