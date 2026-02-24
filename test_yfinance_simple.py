import yfinance as yf
import pandas as pd
import sys

print(f"Python version: {sys.version}")
print(f"yfinance version: {yf.__version__}")

try:
    print("Attempting to download 'AAPL'...")
    data = yf.download("AAPL", period="1mo", progress=False)
    print("Download result type:", type(data))
    print(data.head())
    if data.empty:
        print("Data is empty.")
    else:
        print("Data download successful.")
except Exception as e:
    print(f"Download failed with error: {e}")
