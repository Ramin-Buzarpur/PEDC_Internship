from __future__ import annotations

import numpy as np

from src.data.features import build_features, ewma_vol


def test_ewma_vol_positive_finite(synth_arrays):
    v = ewma_vol(synth_arrays["close_adj"], span=32)
    assert v.shape == synth_arrays["close_adj"].shape
    assert np.all(np.isfinite(v))
    assert np.all(v > 0)


def test_build_features_shapes(synth_arrays, synth_dates, synth_symbols):
    cfg = {"ewma_span": 32, "momentum_horizons": [1, 5, 21], "target_clip": 20.0}
    feats = build_features(synth_dates, synth_symbols, synth_arrays, cfg)
    T, N = synth_arrays["close_adj"].shape
    for name, arr in feats.items():
        assert arr.shape == (T, N), name
        if name != "target_volscaled":
            assert np.isfinite(arr).all(), name
        else:
            assert np.isfinite(arr[:-1]).all(), name
    tgt = feats["target_volscaled"]
    assert np.isnan(tgt[-1]).all()
    assert np.nanmax(np.abs(tgt[:-1])) <= 20.0 + 1e-6


def test_momentum_sign_consistency(synth_arrays, synth_dates, synth_symbols):
    cfg = {"ewma_span": 32, "momentum_horizons": [21]}
    feats = build_features(synth_dates, synth_symbols, synth_arrays, cfg)
    close = synth_arrays["close_adj"]
    up = (close[100] / close[100 - 21] - 1) > 0
    mom = feats["mom_21"][100]
    assert np.sign(mom[up]).sum() >= (up.sum() // 2)
