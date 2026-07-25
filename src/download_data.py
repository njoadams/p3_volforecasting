"""
Downloads historical daily market data from Yahoo Finance
and stores it in the project's raw data directory.
"""

import yfinance as yf
from pathlib import Path

RAW_DIR = Path("data/raw")

RAW_DIR.mkdir(parents=True, exist_ok=True)

ticker = "SPY"
start = "2015-01-01"
end = None

df = yf.download(
    ticker,
    start=start,
    end=end,
    auto_adjust=False,
)
df.to_parquet(RAW_DIR / f"{ticker.lower()}_daily.parquet")
df.to_csv(RAW_DIR / f"{ticker.lower()}_daily.csv")
print(df.head())
print(df.tail())

vix = yf.download(
    "^VIX",
    start=start,
    end=end,
    auto_adjust=False,
)
vix.to_parquet("data/raw/vix_daily.parquet")
vix.to_csv("data/raw/vix_daily.csv")
print(vix.head())
print(vix.tail())

print(f"Downloaded {len(df)} rows of {ticker} data.")