# -*- coding: utf-8 -*-
"""
download_eurusd_hourly.py
==========================
Single-asset hourly OHLCV downloader for EUR/USD (or any Yahoo Finance
forex symbol), rewritten from the old multi-crypto downloader.

IMPORTANT, read before relying on this:
Yahoo Finance's free intraday endpoint hard-caps 60-minute ("1h") data at
~730 days of history. Requesting anything older raises:
    "1h data not available ... must be within the last 730 days"
This is a Yahoo API limit, not something this script can work around.
In practice `period="730d"` returns about 2.8 calendar years for EUR/USD
(~17,000 hourly bars) because forex trades ~24h/5days. That is the
realistic ceiling for free hourly data here — NOT 15 years.

If you need 10-15 years of hourly/minute forex data, you need a different
source (Dukascopy tick data, HistData.com, or a paid provider) and a
separate downloader — ask if you want that built.
"""

import os
import time
import random
from datetime import datetime, timezone

import pandas as pd
import yfinance as yf

# =============================================================================
# USER SETTINGS
# =============================================================================

TICKER = "EURUSD=X"
ASSET_NAME = "EUR_USD"

# Yahoo's real ceiling for 60m bars. Do not raise this — it will just fail.
START_DATE = "2023-12-15"
END_DATE = "2023-12-21"
INTERVAL = "60m"

MAX_RETRIES = 3
BACKOFF_FACTOR = 2

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
os.makedirs(OUTPUT_DIR, exist_ok=True)


def fetch_hourly_with_retry(ticker: str, period: str, interval: str) -> pd.DataFrame:
    last_err = None
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            df = yf.Ticker(ticker).history(period=period, interval=interval, auto_adjust=False)
            if df is not None and not df.empty:
                return df
            last_err = RuntimeError("empty response")
        except Exception as e:
            last_err = e
            print(f"  attempt {attempt}/{MAX_RETRIES} failed: {e}")
        if attempt < MAX_RETRIES:
            sleep_time = (BACKOFF_FACTOR ** (attempt - 1)) + random.uniform(0, 0.5)
            time.sleep(sleep_time)
    raise RuntimeError(f"Could not fetch {ticker}: {last_err}")


def build_eurusd_hourly_dataset(
    ticker: str = TICKER,
    asset_name: str = ASSET_NAME,
    period: str = PERIOD,
    interval: str = INTERVAL,
    output_dir: str = OUTPUT_DIR,
) -> pd.DataFrame:
    print("=" * 70)
    print(f"Downloading hourly OHLCV for {asset_name} ({ticker})")
    print(f"Requested window: {period} at {interval} interval")
    print("=" * 70)

    df = fetch_hourly_with_retry(ticker, period, interval)
    df = df[["Open", "High", "Low", "Close"]].dropna(subset=["Close"])
    df = df[df["Close"] > 0].copy()
    df.index.name = "Datetime"
    df = df.sort_index()

    span_days = (df.index.max() - df.index.min()).days
    print(f"\nFetched {len(df)} hourly bars")
    print(f"Range: {df.index.min()} -> {df.index.max()}  (~{span_days} calendar days, ~{span_days/365.25:.2f} years)")
    if span_days < 365 * 3:
        print(
            "NOTE: this is the real ceiling Yahoo Finance allows for free hourly FX data "
            "(~2.8 years). It is not possible to get 10-15 years of hourly data from this source."
        )

    csv_path = os.path.join(output_dir, f"{asset_name}_hourly.csv")
    parquet_path = os.path.join(output_dir, f"{asset_name}_hourly.parquet")
    df.to_csv(csv_path)
    df.to_parquet(parquet_path)

    print(f"\nSaved CSV:     {csv_path}")
    print(f"Saved Parquet: {parquet_path}")
    print("=" * 70)
    return df


if __name__ == "__main__":
    build_eurusd_hourly_dataset()
