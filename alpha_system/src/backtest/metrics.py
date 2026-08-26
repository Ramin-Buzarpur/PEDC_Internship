from __future__ import annotations

import numpy as np


def zscore(x: np.ndarray, axis: int | None = -1) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    mu = x.mean(axis=axis, keepdims=True)
    sd = x.std(axis=axis, keepdims=True) + 1e-9
    return (x - mu) / sd


def annualized_sharpe(returns: np.ndarray, rf_annual: float = 0.0) -> float:
    r = np.asarray(returns, dtype=np.float64)
    if len(r) < 2 or r.std() == 0:
        return 0.0
    excess = r.mean() * 252 - rf_annual
    return float(excess / (r.std() * np.sqrt(252)))


def sortino(returns: np.ndarray, rf_annual: float = 0.0) -> float:
    r = np.asarray(returns, dtype=np.float64)
    downside = r[r < 0]
    if len(downside) == 0 or downside.std() == 0:
        return 0.0
    return float((r.mean() * 252 - rf_annual) / (downside.std() * np.sqrt(252)))


def max_drawdown(equity: np.ndarray) -> float:
    eq = np.asarray(equity, dtype=np.float64)
    peak = np.maximum.accumulate(eq)
    dd = eq / peak - 1.0
    return float(dd.min())


def cagr(returns: np.ndarray) -> float:
    r = np.asarray(returns, dtype=np.float64)
    if len(r) == 0:
        return 0.0
    total = float(np.prod(1.0 + r))
    years = len(r) / 252.0
    return float(total ** (1.0 / max(years, 1e-9)) - 1.0)


def win_rate(returns: np.ndarray) -> float:
    r = np.asarray(returns, dtype=np.float64)
    if len(r) == 0:
        return 0.0
    return float((r > 0).mean())


def turnover_series(weights: np.ndarray) -> np.ndarray:
    w = np.asarray(weights, dtype=np.float64)
    return np.abs(np.diff(w, axis=0)).sum(axis=1)


def breakeven_cost_bps(port_returns: np.ndarray, weights: np.ndarray) -> float:
    pnls = np.nansum(np.asarray(port_returns, dtype=np.float64))
    lev_turnover = float(turnover_series(weights).sum()) * 2.0
    if lev_turnover <= 0:
        return float("inf")
    return float(pnls / lev_turnover * 1e4)


def full_report(name: str, port_returns: np.ndarray, equity: np.ndarray, weights: np.ndarray | None = None) -> dict:
    rep = {
        "name": name,
        "total_return": float(equity[-1] - 1.0),
        "cagr": cagr(port_returns),
        "sharpe": annualized_sharpe(port_returns),
        "sortino": sortino(port_returns),
        "max_drawdown": max_drawdown(equity),
        "win_rate_daily": win_rate(port_returns),
        "ann_vol": float(np.std(port_returns) * np.sqrt(252)),
    }
    if weights is not None:
        rep["avg_gross_leverage"] = float(np.abs(weights).sum(axis=1).mean())
        rep["avg_turnover_daily"] = float(turnover_series(weights).mean())
        rep["breakeven_cost_bps"] = breakeven_cost_bps(port_returns, weights)
    rep["calmar"] = float(rep["cagr"] / abs(rep["max_drawdown"])) if rep["max_drawdown"] < 0 else 0.0
    return rep
