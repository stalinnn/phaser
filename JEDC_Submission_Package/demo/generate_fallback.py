import pandas as pd
import numpy as np
import os

def generate_correlated_walks(tickers, start_date, end_date, market_type="STOCK"):
    dates = pd.date_range(start=start_date, end=end_date, freq="B")
    n_days = len(dates)
    
    # Base Market Factor
    # Stocks drift up, Crypto drifts up more but with higher vol
    drift = 0.0004 if market_type == "STOCK" else 0.001
    vol = 0.015 if market_type == "STOCK" else 0.04
    
    market_factor = np.cumsum(np.random.normal(drift, vol, n_days))
    
    # Inject Crises (e.g., 2020-03, 2022)
    crisis_2020 = (dates.year == 2020) & (dates.month == 3)
    crisis_2022 = (dates.year == 2022)
    
    prices = {}
    for ticker in tickers:
        noise_vol = 0.02 if market_type == "STOCK" else 0.05
        noise = np.random.normal(0, noise_vol, n_days)
        beta = 0.8 + 0.4 * np.random.rand()
        
        # Core returns
        r = beta * np.diff(market_factor, prepend=0) + noise
        
        # Crisis behavior: Correlation approaches 1 (noise vanishes), Volatility explodes
        # 2020 Crash
        r[crisis_2020] = 3.0 * np.diff(market_factor, prepend=0)[crisis_2020] + 0.1 * noise[crisis_2020]
        # 2022 Bear
        r[crisis_2022] = 1.2 * np.diff(market_factor, prepend=0)[crisis_2022] + 0.5 * noise[crisis_2022]
        
        p = 100 * np.exp(np.cumsum(r))
        prices[ticker] = p
        
    return pd.DataFrame(prices, index=dates)

def main():
    os.makedirs("JEDC_Submission_Package/demo/data", exist_ok=True)
    
    # 1. US SP100 Fallback
    us_tickers = [
        "MSFT", "AMZN", "JPM", "XOM", "GE", "JNJ", "PFE", "C", "KO", "PEP",
        "WMT", "MRK", "INTC", "CSCO", "IBM", "GS", "BAC", "AIG", "AXP", "MCD",
        "AAPL", "GOOGL", "NVDA", "TSLA", "META", "BRK-B", "UNH", "LLY", "V"
    ]
    df_us = generate_correlated_walks(us_tickers, "2000-01-01", "2025-01-01", "STOCK")
    df_us.to_csv("JEDC_Submission_Package/demo/data/US_SP100_fallback.csv")
    print("Generated US fallback data.")

    # 2. Crypto Fallback
    crypto_tickers = [
        "BTC-USD", "ETH-USD", "BNB-USD", "SOL-USD", "XRP-USD", "ADA-USD"
    ]
    df_crypto = generate_correlated_walks(crypto_tickers, "2018-01-01", "2025-01-01", "CRYPTO")
    df_crypto.to_csv("JEDC_Submission_Package/demo/data/CRYPTO_fallback.csv")
    print("Generated Crypto fallback data.")

if __name__ == "__main__":
    main()
