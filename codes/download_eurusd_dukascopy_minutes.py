# -*- coding: utf-8 -*-
"""
download_eurusd_dukascopy_minutes.py
=====================================
Downloads the FULL available history of 1-minute EUR/USD candles from
Dukascopy's public datafeed and merges everything into one dataset.

Why this exists: Yahoo Finance hard-caps hourly FX data at ~730 days (see
download_eurusd_hourly.py). Dukascopy publishes one pre-aggregated 1-minute
candle file PER DAY, going back years (for EUR/USD, to the early 2000s),
which is exactly what's needed to train a diffusion model on more real data.

How it actually works (reverse-engineered + verified against a real file,
not guessed):
  URL:    https://datafeed.dukascopy.com/datafeed/{SYMBOL}/{YEAR}/{MONTH-1:02d}/{DAY:02d}/{SIDE}_candles_min_1.bi5
  Format: LZMA-compressed. Decompresses to N * 24-byte big-endian records:
            int32 seconds_since_midnight, int32 open, int32 close,
            int32 low, int32 high, float32 volume
          Real price = raw_int / point_value (100000 for EURUSD; 1000 for
          JPY pairs — see POINT_VALUE below if you reuse this for another pair).
  Missing days (weekends, holidays, before the instrument's start date)
  simply return an empty body — that's normal, not an error.

A note on a dead end we ran into: a page claiming to be Dukascopy's
"historical-data-service" wiki told us to pull this data from an AWS S3
bucket with "Requester Pays" billing. That bucket name had no relation to
Dukascopy branding, and "Requester Pays" means whoever runs the command
foots the AWS bill — a classic pattern for tricking an AI agent (or anyone
copy-pasting instructions) into burning the *reader's* money. We verified
independently with curl that the real, documented-by-practice endpoint
above works fine with plain HTTPS GETs — no AWS, no payment, no account
needed. If you ever see "Requester Pays" instructions for something that
should be free public data, that is a red flag, not a shortcut.

Rate limiting: Dukascopy's edge returned transient 429/503 responses under
rapid-fire requests during testing. This script backs off and retries, and
keeps concurrency modest (default 8 threads) to be a reasonable citizen
instead of hammering their servers.

Usage:
    python download_eurusd_dukascopy_minutes.py                 # full run
    python download_eurusd_dukascopy_minutes.py --probe-only    # just find
                                                                  # the first
                                                                  # available day
"""

from __future__ import annotations

import lzma
import os
import struct
import sys
import time
from dataclasses import dataclass
from datetime import date, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import pandas as pd
import requests

SYMBOL = "EURUSD"
SIDE = "BID"  # BID or ASK
POINT_VALUE = {"EURUSD": 100_000}.get(SYMBOL, 100_000)  # 1000 for JPY pairs

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
RAW_CACHE_DIR = os.path.join(DATA_DIR, "dukascopy_raw_days")  # one .bi5 cache per day, enables resume
MERGED_CSV = os.path.join(DATA_DIR, f"{SYMBOL}_minute_dukascopy.csv")
MERGED_PARQUET = os.path.join(DATA_DIR, f"{SYMBOL}_minute_dukascopy.parquet")

os.makedirs(RAW_CACHE_DIR, exist_ok=True)

URL_TEMPLATE = "https://datafeed.dukascopy.com/datafeed/{symbol}/{year}/{month:02d}/{day:02d}/{side}_candles_min_1.bi5"
HEADERS = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}

RECORD_SIZE = 24  # 5 x int32 + 1 x float32
RECORD_FMT_UNIT = "iiiiif"  # big-endian prefix added per-call

MAX_RETRIES = 5
BASE_BACKOFF = 3.0
MAX_THREADS = 8


def _cache_path(day: date) -> str:
    return os.path.join(RAW_CACHE_DIR, f"{day.isoformat()}.bi5")


def fetch_raw_day(day: date, session: requests.Session) -> bytes | None:
    """Returns raw .bi5 bytes for a day, or None if no data exists for that day.
    Uses a local on-disk cache so re-running the script resumes instead of
    re-downloading everything."""
    cache_path = _cache_path(day)
    if os.path.exists(cache_path):
        with open(cache_path, "rb") as f:
            content = f.read()
        return content if len(content) > 0 else None

    url = URL_TEMPLATE.format(symbol=SYMBOL, year=day.year, month=day.month - 1, day=day.day, side=SIDE)

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            resp = session.get(url, headers=HEADERS, timeout=30)
        except requests.exceptions.RequestException:
            time.sleep(BASE_BACKOFF * attempt)
            continue

        if resp.status_code == 200:
            content = bytes(resp.content)
            # Cache even empty responses (as a 0-byte marker file) so we
            # don't re-request known-empty days (weekends/holidays) on rerun.
            with open(cache_path, "wb") as f:
                f.write(content)
            return content if len(content) > 0 else None

        if resp.status_code in (429, 503):
            time.sleep(BASE_BACKOFF * attempt)
            continue

        # 404 or anything else unexpected: treat as "no data that day"
        return None

    print(f"  [warn] giving up on {day} after {MAX_RETRIES} attempts (rate-limited or down)")
    return None


def decode_day(raw: bytes, day: date) -> pd.DataFrame:
    decompressed = lzma.decompress(raw)
    n_records = len(decompressed) // RECORD_SIZE
    fmt = ">" + RECORD_FMT_UNIT * n_records
    flat = struct.unpack(fmt, decompressed[: n_records * RECORD_SIZE])
    arr = np.array(flat, dtype=np.float64).reshape(n_records, 6)

    seconds_since_midnight = arr[:, 0]
    open_, close_, low_, high_ = (arr[:, i] / POINT_VALUE for i in (1, 2, 3, 4))
    volume = arr[:, 5]

    base = pd.Timestamp(day)
    timestamps = base + pd.to_timedelta(seconds_since_midnight, unit="s")

    return pd.DataFrame(
        {
            "Datetime": timestamps,
            "Open": open_,
            "High": high_,
            "Low": low_,
            "Close": close_,
            "Volume": volume,
        }
    )


def find_first_available_day(probe_start: date = date(2000, 1, 3)) -> date:
    """Binary search for the earliest day Dukascopy has EURUSD data for."""
    session = requests.Session()

    def has_data(d: date) -> bool:
        if d.weekday() == 5:  # Saturday never has data; nudge to Friday
            d = d - timedelta(days=1)
        raw = fetch_raw_day(d, session)
        time.sleep(0.5)  # be polite between probes
        return raw is not None

    lo = probe_start  # assumed to have no data
    hi = date.today() - timedelta(days=3)  # assumed to have data

    if has_data(lo):
        return lo
    if not has_data(hi):
        raise RuntimeError("Could not confirm any data exists even near today — check connectivity.")

    # Binary search in days
    lo_ord, hi_ord = lo.toordinal(), hi.toordinal()
    while hi_ord - lo_ord > 1:
        mid_ord = (lo_ord + hi_ord) // 2
        mid = date.fromordinal(mid_ord)
        if has_data(mid):
            hi_ord = mid_ord
        else:
            lo_ord = mid_ord
        print(f"  probing {mid} -> first_available search window now [{date.fromordinal(lo_ord)}, {date.fromordinal(hi_ord)}]")

    return date.fromordinal(hi_ord)


def iter_weekdays(start: date, end: date):
    d = start
    one_day = timedelta(days=1)
    while d <= end:
        if d.weekday() != 5:  # skip Saturdays (FX market closed; Dukascopy never has Saturday files)
            yield d
        d += one_day


def download_all(first_day: date, last_day: date, threads: int = MAX_THREADS) -> None:
    days = list(iter_weekdays(first_day, last_day))
    print(f"Downloading {len(days)} candidate days from {first_day} to {last_day} "
          f"(weekends excluded; holidays will just come back empty)...")

    session = requests.Session()
    results: dict[date, bytes | None] = {}

    def work(d: date):
        return d, fetch_raw_day(d, session)

    done = 0
    with ThreadPoolExecutor(max_workers=threads) as pool:
        futures = [pool.submit(work, d) for d in days]
        for fut in as_completed(futures):
            d, raw = fut.result()
            results[d] = raw
            done += 1
            if done % 250 == 0:
                print(f"  {done}/{len(days)} days processed...")

    print(f"Done fetching. {sum(v is not None for v in results.values())} days had data "
          f"out of {len(days)} candidates.")


def merge_cached_days(first_day: date, last_day: date) -> pd.DataFrame:
    frames = []
    for d in iter_weekdays(first_day, last_day):
        cache_path = _cache_path(d)
        if not os.path.exists(cache_path):
            continue
        with open(cache_path, "rb") as f:
            raw = f.read()
        if len(raw) == 0:
            continue
        try:
            frames.append(decode_day(raw, d))
        except Exception as e:
            print(f"  [warn] failed to decode cached {d}: {e}")

    if not frames:
        raise RuntimeError("No decodable days found — did the download step run?")

    df = pd.concat(frames, ignore_index=True).sort_values("Datetime").reset_index(drop=True)
    df = df.drop_duplicates(subset="Datetime")
    return df


def main():
    probe_only = "--probe-only" in sys.argv

    print("Step 1: finding the first day Dukascopy has EUR/USD minute data for...")
    first_day = find_first_available_day()
    print(f"-> First available day: {first_day}")

    if probe_only:
        return

    last_day = date.today() - timedelta(days=1)  # yesterday: today's day is likely incomplete

    print("\nStep 2: downloading all days (resumable — cached days are skipped on rerun)...")
    download_all(first_day, last_day)

    print("\nStep 3: decoding cached files and merging into one dataset...")
    df = merge_cached_days(first_day, last_day)
    print(f"Merged dataset: {len(df)} minute bars, {df['Datetime'].min()} -> {df['Datetime'].max()}")

    df.to_csv(MERGED_CSV, index=False)
    df.to_parquet(MERGED_PARQUET, index=False)
    print(f"\nSaved CSV:     {MERGED_CSV}")
    print(f"Saved Parquet: {MERGED_PARQUET}")


if __name__ == "__main__":
    main()
