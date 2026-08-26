from __future__ import annotations

import numpy as np
import pytest


@pytest.fixture
def synth_arrays():
    rng = np.random.default_rng(7)
    T, N = 320, 6
    drift = rng.normal(0.0003, 0.0002, size=N)
    vol = rng.uniform(0.01, 0.03, size=N)
    rets = rng.normal(drift, vol, size=(T, N))
    close = 50.0 * np.cumprod(1 + rets, axis=0)
    high = close * (1 + np.abs(rng.normal(0, 0.005, size=close.shape)))
    low = close * (1 - np.abs(rng.normal(0, 0.005, size=close.shape)))
    open_ = np.vstack([close[0:1], close[:-1]]) * (1 + rng.normal(0, 0.003, size=close.shape))
    volume = rng.lognormal(14, 0.5, size=close.shape).astype(np.float32)
    return {
        "open": open_.astype(np.float32),
        "high": high.astype(np.float32),
        "low": low.astype(np.float32),
        "close_adj": close.astype(np.float32),
        "volume": volume,
    }


@pytest.fixture
def synth_dates(synth_arrays):
    import pandas as pd

    return pd.bdate_range("2020-01-01", periods=synth_arrays["close_adj"].shape[0])


@pytest.fixture
def synth_symbols(synth_arrays):
    return [f"S{i}" for i in range(synth_arrays["close_adj"].shape[1])]
