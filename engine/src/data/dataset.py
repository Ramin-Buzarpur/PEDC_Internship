from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import Dataset

CHANNEL_NAMES = ["ch_open", "ch_high", "ch_low", "ch_close", "vol_n", "sentiment"]


def build_channels(arrays: dict[str, np.ndarray], sentiment: np.ndarray | None) -> np.ndarray:
    close = arrays["close_adj"]
    open_ = arrays["open"]
    high = arrays["high"]
    low = arrays["low"]
    vol = arrays["volume"]
    eps = 1e-6

    c_prev = np.vstack([close[0:1], close[:-1]])
    o_p = np.vstack([open_[0:1], open_[:-1]])
    h_p = np.vstack([high[0:1], high[:-1]])
    l_p = np.vstack([low[0:1], low[:-1]])
    ch_open = (o_p - c_prev) / (np.abs(c_prev) + eps)
    ch_high = (h_p - l_p) / (np.abs(l_p) + eps)
    ch_low = (low - c_prev) / (np.abs(c_prev) + eps)
    ch_close = (close - c_prev) / (np.abs(c_prev) + eps)

    log_vol = np.log1p(np.maximum(vol, 0.0))
    med_vol = np.median(log_vol, axis=0, keepdims=True)
    std_vol = np.std(log_vol, axis=0, keepdims=True) + 1e-3
    vol_n = (log_vol - med_vol) / std_vol

    chans = [ch_open, ch_high, ch_low, ch_close, vol_n]
    if sentiment is not None and sentiment.shape == close.shape:
        chans.append(sentiment.astype(np.float32))
    else:
        chans.append(np.zeros_like(ch_close))
    return np.stack(chans, axis=-1).astype(np.float32)


class StockDiffusionDataset(Dataset):
    def __init__(
        self,
        arrays: dict[str, np.ndarray],
        sentiment: np.ndarray | None,
        start_idx: int,
        end_idx: int,
        seq_len_past: int = 30,
        seq_len_future: int = 10,
        adaptive_alpha: float = 0.7,
        precomputed_weights: np.ndarray | None = None,
    ):
        self.L = seq_len_past
        self.H = seq_len_future
        self.T = self.L + self.H
        self.data = build_channels(arrays, sentiment)

        self.start = start_idx
        self.end = end_idx
        self.num_samples = end_idx - start_idx - self.T + 1
        assert self.num_samples > 0, "not enough data for requested window"

        if precomputed_weights is not None:
            self.weights = precomputed_weights
        else:
            variances = np.zeros(self.num_samples, dtype=np.float64)
            ch_close = self.data[:, :, 3]
            for i in range(self.num_samples):
                w = ch_close[start_idx + i : start_idx + i + self.T]
                variances[i] = np.nanmean(np.nanvar(w, axis=0)) + 1e-9
            med = np.median(variances)
            w = np.sqrt(variances / (med + 1e-12))
            self.weights = np.clip(w, 0.5, 2.0).astype(np.float32)

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        s = self.start + idx
        window = self.data[s : s + self.T]
        mean = window[: self.L].mean(axis=(0, 1), keepdims=True)
        std = window[: self.L].std(axis=(0, 1), keepdims=True) + 1e-5
        norm = (window - mean) / std
        x_all = torch.from_numpy(norm.transpose(2, 1, 0).copy())
        return x_all, torch.tensor(self.weights[idx])


def build_splits(num_days: int, val_ratio: float = 0.15, test_ratio: float = 0.15, purge_gap: int = 40):
    test_start = int(num_days * (1 - test_ratio))
    val_start = int(test_start - num_days * val_ratio - purge_gap)
    return {
        "train": (0, val_start - purge_gap),
        "val": (val_start, test_start),
        "test": (test_start, num_days),
    }
