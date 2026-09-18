from __future__ import annotations

import numpy as np

from src.backtest.engine import BacktestEngine, signals_to_weights
from src.backtest.metrics import annualized_sharpe, breakeven_cost_bps, full_report, max_drawdown, turnover_series, zscore


def test_sharpe_known_series():
    rng = np.random.default_rng(0)
    r2 = rng.normal(0.001, 0.01, size=252 * 10)
    sr = annualized_sharpe(r2)
    expected = r2.mean() * 252 / (r2.std() * np.sqrt(252))
    assert abs(sr - expected) < 1e-9
    assert annualized_sharpe(np.zeros(10)) == 0.0


def test_max_drawdown():
    eq = np.array([1.0, 1.2, 0.6, 0.9])
    assert abs(max_drawdown(eq) - (0.6 / 1.2 - 1)) < 1e-12


def test_engine_no_cost_matches_weighted_returns():
    rng = np.random.default_rng(3)
    T, N = 100, 4
    rets = rng.normal(0, 0.01, size=(T, N))
    w = np.zeros((T, N))
    w[:, :] = 0.25
    net, equity = BacktestEngine(cost_bps=0.0).run(w, rets)
    expected_gross = (w[: T - 1] * rets[1:]).sum(axis=1)
    assert np.allclose(net, expected_gross, atol=1e-12)
    assert np.allclose(equity[1:], np.cumprod(1 + expected_gross))


def test_engine_costs_reduce_returns():
    rng = np.random.default_rng(3)
    T, N = 50, 3
    rets = rng.normal(0, 0.01, size=(T, N))
    w = rng.uniform(-1, 1, size=(T, N)) * 0.3
    net_free, _ = BacktestEngine(cost_bps=0.0).run(w, rets)
    net_cost, _ = BacktestEngine(cost_bps=10.0).run(w, rets)
    assert (net_cost <= net_free + 1e-15).all()


def test_signals_to_weights_caps_and_leverage():
    sig = np.array([[10.0, -10.0, 0.001]])
    vol = np.array([[0.02, 0.01, 0.03]])
    w = signals_to_weights(sig, vol, vol_target_daily=0.01, max_weight=0.25, max_leverage=1.0, min_abs_signal=0.01)
    assert np.abs(w).max() <= 0.25 + 1e-8
    assert np.abs(w).sum() <= 1.0 + 1e-8
    assert w[0, 2] == 0.0


def test_turnover_and_breakeven():
    w = np.zeros((5, 2))
    w[1] = [0.5, -0.5]
    w[3] = [-0.5, 0.5]
    to = turnover_series(w)
    assert abs(to[0] - 1.0) < 1e-12
    rets = np.full((4,), 0.001)
    be = breakeven_cost_bps(rets, w)
    assert np.isfinite(be)


def test_zscore_and_report():
    z = zscore(np.array([1.0, 2.0, 3.0]))
    assert abs(z.mean()) < 1e-9 and abs(z.std() - 1) < 1e-8
    r = np.random.default_rng(1).normal(0.0005, 0.008, 500)
    eq = np.cumprod(1 + r)
    rep = full_report("t", r, eq, np.zeros((len(r), 2)))
    for k in ["sharpe", "cagr", "max_drawdown", "win_rate_daily"]:
        assert k in rep
