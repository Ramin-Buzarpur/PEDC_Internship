from __future__ import annotations

import numpy as np
import pandas as pd


def _df(arr: np.ndarray, dates, syms) -> pd.DataFrame:
    return pd.DataFrame(arr, index=dates, columns=syms)


def ewma_vol(close_adj: np.ndarray, span: int = 32) -> np.ndarray:
    rets = pd.DataFrame(close_adj).pct_change(fill_method=None)
    daily_vol = rets.ewm(span=span, adjust=False).std()
    out = (daily_vol * np.sqrt(252.0)).bfill().ffill()
    return out.fillna(0.05).clip(lower=0.005).values.astype(np.float32)


def build_features(dates, syms, arrays: dict[str, np.ndarray], cfg: dict) -> dict[str, np.ndarray]:
    span = int(cfg.get("ewma_span", 32))
    horizons = list(cfg.get("momentum_horizons", [1, 5, 21, 63]))
    fast, slow, sig = (int(cfg.get("macd_fast", 12)), int(cfg.get("macd_slow", 26)), int(cfg.get("macd_signal", 9)))
    clip = float(cfg.get("target_clip", 20.0))

    close = _df(arrays["close_adj"], dates, syms)
    high = _df(arrays["high"], dates, syms)
    low = _df(arrays["low"], dates, syms)
    opn = _df(arrays["open"], dates, syms)
    vol = _df(arrays["volume"], dates, syms)

    feats: dict[str, np.ndarray] = {}
    sigma = ewma_vol(arrays["close_adj"], span)
    sigma_df = _df(sigma, dates, syms)

    r1 = close.pct_change(fill_method=None)
    feats["ret_1d"] = r1.fillna(0.0).values.astype(np.float32)

    for h in horizons:
        mom_h = close.pct_change(h, fill_method=None)
        normed = mom_h / ((sigma_df / np.sqrt(252.0)) * np.sqrt(max(h, 1)) + 1e-8)
        feats[f"mom_{h}"] = normed.clip(-20, 20).bfill().ffill().fillna(0.0).values.astype(np.float32)

    ema_f = close.ewm(span=fast, adjust=False).mean()
    ema_s = close.ewm(span=slow, adjust=False).mean()
    macd = ema_f - ema_s
    q = macd / (close.rolling(63, min_periods=10).std() + 1e-8)
    signal = q / (q.rolling(252, min_periods=21).std() + 1e-8)
    feats["macd_signal"] = signal.clip(-20, 20).bfill().ffill().fillna(0.0).values.astype(np.float32)

    prev_close = close.shift(1)
    feats["range"] = ((high - low) / (prev_close + 1e-8)).clip(0, 5).fillna(0.0).values.astype(np.float32)
    feats["gap"] = ((opn - prev_close) / (prev_close + 1e-8)).clip(-2, 2).fillna(0.0).values.astype(np.float32)
    feats["body"] = (((close - opn) / (opn + 1e-8)) / (sigma_df / np.sqrt(252.0) + 1e-8)).clip(-20, 20).fillna(0.0).values.astype(np.float32)

    v_mu = vol.rolling(20, min_periods=5).mean()
    v_sd = vol.rolling(20, min_periods=5).std()
    feats["vol_z"] = ((vol - v_mu) / (v_sd + 1e-8)).clip(-10, 10).bfill().ffill().fillna(0.0).values.astype(np.float32)

    feats["ewma_vol"] = sigma.astype(np.float32)

    fwd_ret = close.shift(-1) / close - 1.0
    tgt = fwd_ret / ((sigma_df / np.sqrt(252.0)) + 1e-8)
    feats["target_volscaled"] = tgt.clip(-clip, clip).values.astype(np.float32)

    return feats


FEATURE_ORDER = ["ret_1d", "mom_1", "mom_5", "mom_21", "mom_63", "macd_signal", "range", "gap", "body", "vol_z", "sentiment"]
