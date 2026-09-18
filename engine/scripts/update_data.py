from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def main():
    ap = argparse.ArgumentParser(description="Extend the cached price panel with fresh data from Yahoo Finance (best-effort)")
    ap.add_argument("--days", type=int, default=10, help="how many calendar days back to fetch")
    ap.add_argument("--batch", type=int, default=12)
    args = ap.parse_args()

    try:
        import yfinance as yf
    except ImportError:
        print("yfinance not installed: pip install yfinance")
        return 1

    cache_dir = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data_cache"))
    data_path = os.path.join(cache_dir, "panels.npz")
    meta_path = os.path.join(cache_dir, "universe_meta.json")
    if not (os.path.exists(data_path) and os.path.exists(meta_path)):
        print("cache not found; run scripts/prepare_cache.py first")
        return 1

    z = np.load(data_path)
    dates = pd.to_datetime([d.decode() if isinstance(d, bytes) else d for d in z["dates"]])
    symbols = [s.decode() if isinstance(s, bytes) else s for s in z["symbols"]]
    last = dates[-1]
    start = (last + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
    print(f"cache ends {last.date()}; fetching {start} → today for {len(symbols)} symbols")

    ref = None
    for i in range(0, len(symbols), args.batch):
        try:
            ref = yf.download(symbols[i : i + args.batch], start=start, progress=False, auto_adjust=False, group_by="column", threads=True)
        except Exception as e:
            print(f"batch {i} failed: {e}")
            ref = None
        if ref is not None and len(ref) > 0:
            break
        time.sleep(1.0)
    if ref is None or len(ref) == 0:
        print("Yahoo Finance unavailable (rate limit / network). Cache unchanged. Try again later.")
        return 1

    new_idx = pd.to_datetime(ref.index).normalize()
    T_new = len(new_idx)
    arrays = {k: z[k] for k in ["open", "high", "low", "close_adj", "volume"]}
    T_old, N = arrays["close_adj"].shape
    new_open = np.full((T_new, N), np.nan, dtype=np.float32)
    new_high = np.full((T_new, N), np.nan, dtype=np.float32)
    new_low = np.full((T_new, N), np.nan, dtype=np.float32)
    new_close = np.full((T_new, N), np.nan, dtype=np.float32)
    new_vol = np.full((T_new, N), np.nan, dtype=np.float32)

    for i in range(0, len(symbols), args.batch):
        chunk = symbols[i : i + args.batch]
        try:
            df = yf.download(chunk, start=start, progress=False, auto_adjust=False, group_by="column", threads=True)
        except Exception:
            continue
        if df is None or len(df) == 0:
            continue
        idx = pd.to_datetime(df.index).normalize()
        pos = {d: k for k, d in enumerate(new_idx)}
        if isinstance(df.columns, pd.MultiIndex):
            for sym in chunk:
                if sym not in df.columns.get_level_values(1):
                    continue
                j = symbols.index(sym)
                sub = df.xs(sym, axis=1, level=1)
                for d, row in sub.iterrows():
                    k = pos.get(pd.Timestamp(d).normalize())
                    if k is None or pd.isna(row.get("Close")):
                        continue
                    adj = row.get("Adj Close", row["Close"])
                    new_open[k, j] = row.get("Open", np.nan)
                    new_high[k, j] = row.get("High", np.nan)
                    new_low[k, j] = row.get("Low", np.nan)
                    new_close[k, j] = adj if not pd.isna(adj) else row["Close"]
                    new_vol[k, j] = row.get("Volume", np.nan)
        else:
            sym = chunk[0]
            j = symbols.index(sym)
            for d, row in df.iterrows():
                k = pos.get(pd.Timestamp(d).normalize())
                if k is None or pd.isna(row.get("Close")):
                    continue
                adj = row.get("Adj Close", row["Close"])
                new_open[k, j] = row.get("Open", np.nan)
                new_high[k, j] = row.get("High", np.nan)
                new_low[k, j] = row.get("Low", np.nan)
                new_close[k, j] = adj if not pd.isna(adj) else row["Close"]
                new_vol[k, j] = row.get("Volume", np.nan)
        time.sleep(1.0)

    valid = ~np.all(np.isnan(new_close), axis=1)
    if not valid.any():
        print("no valid new rows; cache unchanged")
        return 1

    backup = data_path + ".bak"
    shutil.copy2(data_path, backup)
    out = {}
    for key, old, add in [
        ("open", arrays["open"], new_open[valid]),
        ("high", arrays["high"], new_high[valid]),
        ("low", arrays["low"], new_low[valid]),
        ("close_adj", arrays["close_adj"], new_close[valid]),
        ("volume", arrays["volume"], new_vol[valid]),
    ]:
        merged = np.vstack([old, add])
        for j in range(merged.shape[1]):
            col = pd.Series(merged[:, j]).ffill().bfill().values
            merged[:, j] = col
        out[key] = merged.astype(np.float32)
    all_dates = dates.append(pd.DatetimeIndex(new_idx[valid]))
    np.savez_compressed(data_path, dates=np.array([d.strftime("%Y-%m-%d") for d in all_dates]), symbols=np.array(symbols), **out)
    print(f"cache updated: +{int(valid.sum())} sessions → ends {all_dates[-1].date()} (backup at {backup})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
