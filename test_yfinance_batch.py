import yfinance as yf

tickers = ["AAPL", "MSFT", "GOOGL"]
print(f"yfinance version: {yf.__version__}")
print("Downloading batch...")
try:
    data = yf.download(tickers, start="2023-01-01", end="2023-01-10", progress=False, threads=False)
    print("Shape:", data.shape)
    print(data.head())
except Exception as e:
    print(f"Batch download failed: {e}")
