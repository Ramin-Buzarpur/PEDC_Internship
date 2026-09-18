from __future__ import annotations

import glob
import json
import os

import numpy as np
import pandas as pd


REQUIRED_COLS = ["Date", "Open", "High", "Low", "Close", "Volume"]
OPTIONAL_COLS = ["Adjusted Close"]


def _read_csv(path: str) -> pd.DataFrame | None:
    try:
        df = pd.read_csv(path, usecols=lambda c: c.strip() in REQUIRED_COLS + OPTIONAL_COLS)
        df.columns = [c.strip() for c in df.columns]
    except Exception:
        return None
    date_col = next((c for c in ["Date", "date", "Timestamp", "time"] if c in df.columns), None)
    if date_col is None or len(df) < 60:
        return None
    dt = pd.to_datetime(df[date_col], dayfirst=True, errors="coerce")
    valid = dt.notna()
    if valid.sum() < 60:
        return None
    df = df[valid].copy()
    df.index = dt[valid]
    df.index.name = "Date"
    df = df[~df.index.duplicated(keep="last")].sort_index()
    price_cols = [c for c in ["Open", "High", "Low", "Close"] if c in df.columns]
    if len(price_cols) < 4:
        return None
    for c in price_cols + ["Volume"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    if "Adjusted Close" in df.columns:
        df["Adjusted Close"] = pd.to_numeric(df["Adjusted Close"], errors="coerce")
        bad = df["Adjusted Close"].isna() & df["Close"].notna()
        df.loc[bad, "Adjusted Close"] = df.loc[bad, "Close"]
    else:
        df["Adjusted Close"] = df["Close"]
    df = df.dropna(subset=["Close", "Volume"])
    if len(df) < 60:
        return None
    return df[["Open", "High", "Low", "Close", "Adjusted Close", "Volume"]]


def scan_universe(data_dir: str, markets: list[str]) -> list[tuple[str, str]]:
    seen: set[str] = set()
    universe: list[tuple[str, str]] = []
    for mkt in markets:
        csv_dir = os.path.join(data_dir, mkt, "csv")
        if not os.path.isdir(csv_dir):
            continue
        files = sorted(glob.glob(os.path.join(csv_dir, "*.csv"))) + sorted(glob.glob(os.path.join(csv_dir, "*.CSV")))
        for f in files:
            sym = os.path.splitext(os.path.basename(f))[0].strip().upper()
            if sym and sym not in seen:
                seen.add(sym)
                universe.append((sym, f))
    return universe


def select_liquid_symbols(
    universe: list[tuple[str, str]],
    max_stocks: int,
    min_history_days: int,
    progress_every: int = 500,
    log=print,
) -> tuple[list[str], list[str], dict[str, float]]:
    scores: list[tuple[float, str, str]] = []
    for i, (sym, path) in enumerate(universe):
        df = _read_csv(path)
        if df is None or len(df) < min_history_days:
            continue
        tail = df.tail(90)
        liq = float((tail["Close"] * tail["Volume"]).mean())
        if liq <= 0:
            continue
        span_days = (df.index[-1] - df.index[0]).days
        recency_bonus = 1.0 if (pd.Timestamp.today() - df.index[-1]).days < 30 else 0.5
        score = liq * recency_bonus * np.log1p(span_days / 365.0)
        scores.append((score, sym, path))
        if (i + 1) % progress_every == 0:
            log(f"scanned {i + 1}/{len(universe)} files, {len(scores)} candidates")
    scores.sort(reverse=True)
    chosen = scores[:max_stocks]
    syms = [s for _, s, _ in chosen]
    paths = [p for _, _, p in chosen]
    liq_map = {s: sc for sc, s, _ in chosen}
    return syms, paths, liq_map


def build_panels(paths: list[str], syms: list[str]) -> dict[str, pd.DataFrame]:
    frames = {}
    for sym, path in zip(syms, paths):
        df = _read_csv(path)
        if df is None:
            continue
        frames[sym] = df
    idx = sorted(set().union(*[set(f.index) for f in frames.values()]))
    idx = pd.DatetimeIndex(idx).normalize()
    panels = {}
    for col in ["Open", "High", "Low", "Close", "Adjusted Close", "Volume"]:
        mat = pd.DataFrame(index=idx, columns=list(frames.keys()), dtype=float)
        for sym, f in frames.items():
            mat.loc[f.index, sym] = f[col].values
        mat = mat.sort_index().ffill().bfill()
        panels[col] = mat.astype(np.float32)
    first_valid = panels["Close"].notna().all(axis=1).idxmax()
    panels = {k: v.loc[first_valid:] for k, v in panels.items()}
    return panels


def load_market_data(
    raw_dir: str,
    cache_dir: str,
    markets: list[str],
    max_stocks: int,
    min_history_days: int,
    force_rescan: bool = False,
    log=print,
) -> tuple[pd.DatetimeIndex, list[str], dict[str, np.ndarray]]:
    os.makedirs(cache_dir, exist_ok=True)
    meta_path = os.path.join(cache_dir, "universe_meta.json")
    data_path = os.path.join(cache_dir, "panels.npz")
    if not force_rescan and os.path.exists(meta_path) and os.path.exists(data_path):
        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)
        if meta.get("max_stocks") == max_stocks:
            z = np.load(data_path, allow_pickle=False)
            dates = pd.to_datetime([d.decode() if isinstance(d, bytes) else d for d in z["dates"]])
            symbols = [s.decode() if isinstance(s, bytes) else s for s in z["symbols"]]
            arrays = {k: z[k] for k in ["open", "high", "low", "close_adj", "volume"]}
            log(f"loaded cached panels: T={len(dates)}, N={len(symbols)}")
            return dates, symbols, arrays

    log("scanning universe across markets ...")
    universe = scan_universe(raw_dir, markets)
    log(f"found {len(universe)} unique symbols")
    syms, paths, _ = select_liquid_symbols(universe, max_stocks, min_history_days, log=log)
    if len(syms) == 0:
        raise RuntimeError("no eligible stocks found; check raw_dir")
    log(f"selected {len(syms)} most liquid symbols: {syms}")
    panels = build_panels(paths, syms)

    dates = panels["Close"].index
    arrays = {
        "open": panels["Open"].values,
        "high": panels["High"].values,
        "low": panels["Low"].values,
        "close_adj": panels["Adjusted Close"].values,
        "volume": panels["Volume"].values,
    }
    np.savez_compressed(
        data_path,
        dates=np.array([d.strftime("%Y-%m-%d") for d in dates]),
        symbols=np.array(syms),
        **arrays,
    )
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump({"max_stocks": max_stocks, "markets": markets, "symbols": syms}, f)
    log(f"cached panels to {data_path}: T={len(dates)} days x N={len(syms)} stocks")
    return dates, syms, arrays
