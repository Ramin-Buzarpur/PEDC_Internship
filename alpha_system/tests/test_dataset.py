from __future__ import annotations

import numpy as np

from src.data.dataset import StockDiffusionDataset, build_channels, build_splits


def test_build_splits_purge():
    s = build_splits(1000, val_ratio=0.15, test_ratio=0.15, purge_gap=40)
    assert s["train"][1] + 40 <= s["val"][0]
    assert s["val"][1] <= s["test"][0]
    assert s["test"][1] == 1000
    assert all(a < b for a, b in s.values())


def test_build_channels_shape_and_finiteness(synth_arrays):
    data = build_channels(synth_arrays, None)
    T, N = synth_arrays["close_adj"].shape
    assert data.shape == (T, N, 6)
    assert np.isfinite(data).all()
    sent = np.random.default_rng(0).uniform(-1, 1, size=(T, N)).astype(np.float32)
    data2 = build_channels(synth_arrays, sent)
    assert np.allclose(data2[..., 5], sent)


def test_dataset_normalization_stats_from_past_only(synth_arrays, synth_symbols=None):
    ds = StockDiffusionDataset(synth_arrays, None, 50, 200, seq_len_past=30, seq_len_future=10)
    x, w = ds[10]
    past = x[:, :, :30].numpy()
    assert abs(float(past.mean())) < 5.0
    assert float(past.std()) > 0
    N = synth_arrays["close_adj"].shape[1]
    assert x.shape == (6, N, 40)
    assert 0.5 <= w.item() <= 2.0


def test_dataset_len_matches(synth_arrays):
    ds = StockDiffusionDataset(synth_arrays, None, 50, 200, 30, 10)
    assert len(ds) == 200 - 50 - 40 + 1
