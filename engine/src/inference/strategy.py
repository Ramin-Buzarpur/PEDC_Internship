from __future__ import annotations

import numpy as np
import torch

from ..backtest.metrics import zscore


def combine_signals(
    rl_signal: np.ndarray | None,
    diffusion_signal: np.ndarray,
    momentum_z: np.ndarray,
    w_rl: float = 0.45,
    w_diff: float = 0.35,
    w_mom: float = 0.20,
) -> np.ndarray:
    d_sig = np.nan_to_num(zscore(diffusion_signal))
    m_sig = np.nan_to_num(zscore(momentum_z))
    if rl_signal is None:
        total = w_diff + w_mom
        return (w_diff * d_sig + w_mom * m_sig) / total
    r_sig = np.clip(np.asarray(rl_signal, dtype=np.float64), -1, 1)
    return w_rl * r_sig + w_diff * d_sig + w_mom * m_sig
