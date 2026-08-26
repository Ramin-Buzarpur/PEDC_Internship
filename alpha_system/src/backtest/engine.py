from __future__ import annotations

import numpy as np


def signals_to_weights(
    signals: np.ndarray,
    daily_vol: np.ndarray,
    vol_target_daily: float = 0.15 / np.sqrt(252),
    max_weight: float = 0.25,
    max_leverage: float = 1.0,
    min_abs_signal: float = 0.0,
) -> np.ndarray:
    sig = np.where(np.abs(signals) < min_abs_signal, 0.0, signals)
    w = sig * (vol_target_daily / np.maximum(daily_vol, 1e-5))
    w = np.clip(w, -max_weight, max_weight)
    gross = np.abs(w).sum(axis=1, keepdims=True)
    scale = np.minimum(1.0, max_leverage / np.maximum(gross, 1e-9))
    return (w * scale).astype(np.float32)


def core_trend_weights(
    daily_vol: np.ndarray,
    bull_mask: np.ndarray,
    vol_target_daily: float,
    max_weight: float,
    core_gross: float,
    bear_multiplier: float = 0.3,
) -> np.ndarray:
    T, N = daily_vol.shape
    w = np.where(bull_mask[:, None], 1.0, bear_multiplier)
    raw = w * (vol_target_daily / np.maximum(daily_vol, 1e-5))
    raw = np.clip(raw, 0.0, max_weight)
    gross = raw.sum(axis=1, keepdims=True)
    scale = np.minimum(1.0, core_gross / np.maximum(gross, 1e-9))
    return (raw * scale).astype(np.float32)


def combine_core_satellite(core_w: np.ndarray, alpha_w: np.ndarray, satellite_share: float) -> np.ndarray:
    s = float(np.clip(satellite_share, 0.0, 1.0))
    return ((1.0 - s) * core_w + s * alpha_w).astype(np.float32)


class BacktestEngine:
    def __init__(self, cost_bps: float = 5.0):
        self.cost = float(cost_bps) / 1e4

    def run(self, weights: np.ndarray, returns: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        T, N = returns.shape
        w = weights[: T - 1]
        r_next = returns[1:]
        gross = (w * r_next).sum(axis=1)
        turnover = np.abs(np.diff(weights, axis=0)).sum(axis=1)[: T - 1]
        net = gross - turnover * self.cost
        equity = np.cumprod(1.0 + net)
        equity = np.concatenate([[1.0], equity])
        return net, equity
