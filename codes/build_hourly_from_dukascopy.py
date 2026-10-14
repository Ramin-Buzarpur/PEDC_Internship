# -*- coding: utf-8 -*-
"""
build_hourly_from_dukascopy.py
===============================
Resamples the full-history 1-minute Dukascopy EUR/USD dataset
(data/EURUSD_minute_dukascopy.parquet, 2003-05-04 -> now) into hourly
OHLC bars, in the same CSV schema as EUR_USD_hourly.csv (Datetime,Open,
High,Low,Close), so train_diffusion_eurusd.py can be pointed at it.

Only minutes with Volume > 0 (actual trades) are used to build each hourly
bar; hours with zero traded volume (weekends/holidays) are dropped
entirely rather than resampled into fake flat candles — this matches how
the original Yahoo-sourced hourly dataset behaves (irregular bars/day,
handled downstream via the median bars_per_day logic).

Usage:
    python build_hourly_from_dukascopy.py
"""

import os

import pandas as pd

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
MINUTE_PARQUET = os.path.join(DATA_DIR, "EURUSD_minute_dukascopy.parquet")
OUT_CSV = os.path.join(DATA_DIR, "EUR_USD_hourly_full.csv")
OUT_PARQUET = os.path.join(DATA_DIR, "EUR_USD_hourly_full.parquet")


def build() -> pd.DataFrame:
    print(f"Loading {MINUTE_PARQUET} ...")
    df = pd.read_parquet(MINUTE_PARQUET)
    df["Datetime"] = pd.to_datetime(df["Datetime"])
    df = df[df["Volume"] > 0].copy()
    df = df.set_index("Datetime").sort_index()

    bucket = df.index.floor("h")
    hourly = df.groupby(bucket).agg(
        Open=("Open", "first"),
        High=("High", "max"),
        Low=("Low", "min"),
        Close=("Close", "last"),
    )
    hourly.index.name = "Datetime"

    span_days = (hourly.index.max() - hourly.index.min()).days
    print(f"Hourly bars: {len(hourly)}  range: {hourly.index.min()} -> {hourly.index.max()}  "
          f"(~{span_days/365.25:.1f} years)")

    hourly.to_csv(OUT_CSV)
    hourly.to_parquet(OUT_PARQUET)
    print(f"Saved: {OUT_CSV}\nSaved: {OUT_PARQUET}")
    return hourly


if __name__ == "__main__":
    build()
