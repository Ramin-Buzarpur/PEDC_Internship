from __future__ import annotations

import numpy as np


def build_relation_matrix(close_adj: np.ndarray, start: int, end: int, corr_window: int, threshold: float) -> np.ndarray:
    window = close_adj[start:end]
    rets = window[1:] / (window[:-1] + 1e-9) - 1.0
    if rets.shape[0] < 20:
        n = close_adj.shape[1]
        return np.tile(np.eye(n, dtype=np.float32)[..., None], (1, 1, 2))
    rets = np.clip(rets, -0.5, 0.5)
    corr = np.corrcoef(rets.T)
    corr = np.nan_to_num(corr, nan=0.0)
    n = corr.shape[0]
    rel = np.zeros((n, n, 2), dtype=np.float32)
    rel[:, :, 0] = (corr > threshold).astype(np.float32)
    rel[:, :, 1] = (corr < -threshold).astype(np.float32)
    idx = np.arange(n)
    rel[idx, idx, :] = 1.0
    return rel
