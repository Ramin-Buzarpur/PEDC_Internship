from __future__ import annotations

import time

import numpy as np
import pandas as pd
import torch

from ..backtest.engine import BacktestEngine, combine_core_satellite, core_trend_weights, signals_to_weights
from ..backtest.metrics import full_report, zscore
from ..data.dataset import build_splits
from ..inference.predictor import ScenarioPredictor
from ..inference.strategy import combine_signals


def build_bull_mask(arrays: dict[str, np.ndarray], trend_window: int = 200) -> np.ndarray:
    mkt_idx = np.nanmean(arrays["close_adj"], axis=1)
    ma = pd.Series(mkt_idx).rolling(trend_window, min_periods=max(20, trend_window // 4)).mean().values
    with np.errstate(invalid="ignore"):
        bull = (mkt_idx > ma).astype(np.float32)
    bull[np.isnan(ma)] = 1.0
    return bull


def smooth_signal(sig: np.ndarray, span: int) -> np.ndarray:
    if span <= 1:
        return sig
    a = 2.0 / (span + 1.0)
    out = np.empty_like(sig)
    out[0] = sig[0]
    for t in range(1, len(sig)):
        out[t] = a * sig[t] + (1 - a) * out[t - 1]
    return out


def run_walkforward(
    cfg,
    dates,
    symbols,
    arrays,
    sentiment,
    feats,
    diffusion_art: dict,
    rl_agent=None,
    log=print,
):
    bcfg = cfg.section("backtest")
    scfg = cfg.section("strategy")
    rcfg = cfg.section("rl")
    tcfg = cfg.section("training")
    dcfg = cfg.section("diffusion")
    device = next(diffusion_art["diffusion"].model.parameters()).device

    T_days = arrays["close_adj"].shape[0]
    splits = build_splits(T_days, float(tcfg.get("val_ratio", 0.15)), 0.15, int(tcfg.get("purge_gap", 40)))
    test_start, test_end = splits["test"]
    L = int(cfg.get("dataset.seq_len_past", 30))
    H = int(cfg.get("dataset.seq_len_future", 10))
    N = len(symbols)

    predictor = ScenarioPredictor(diffusion_art["diffusion"], arrays, sentiment, L, H, device)
    rel = diffusion_art["rel_matrix"]

    returns_daily = (arrays["close_adj"][1:] / arrays["close_adj"][:-1] - 1.0).astype(np.float32)
    returns_daily = np.vstack([np.zeros((1, N), dtype=np.float32), returns_daily])
    daily_vol = np.maximum(feats["ewma_vol"] / np.sqrt(252.0), 1e-5)

    infer_env = None
    base_signal = None
    resid_scale = 1.0
    if rl_agent is not None:
        from ..backtest.metrics import zscore
        from ..models.rl_agent import PortfolioEnv
        from ..training.train_rl import build_state_feats

        per_asset = (rl_agent.state_dim - 2) // len(symbols) - 2
        state_feats_full = build_state_feats(feats, sentiment, target_channels=per_asset)
        base_signal = np.clip(zscore(feats["mom_21"]), -3.0, 3.0) / 3.0
        resid_scale = float(rcfg.get("resid_scale", 1.0))
        infer_env = PortfolioEnv(
            returns=returns_daily,
            state_feats=state_feats_full,
            daily_vol=daily_vol,
            start=test_start,
            end=test_end,
            vol_target_annual=float(rcfg.get("vol_target_annual", 0.15)),
            max_leverage=float(rcfg.get("max_leverage", 1.0)),
            max_weight=float(rcfg.get("max_weight", 0.25)),
            cost_bps=float(rcfg.get("cost_bps", 5.0)),
            drawdown_penalty=float(rcfg.get("drawdown_penalty", 2.0)),
            base_signal=base_signal,
            resid_scale=resid_scale,
        )

    signals = np.zeros((T_days, N), dtype=np.float32)
    refresh = max(1, int(dcfg.get("refresh_every", 5)))
    diff_sig_cache = None
    live_weights = np.zeros(N, dtype=np.float32)
    live_dd = 0.0
    live_equity = [1.0]
    t0 = time.time()
    n_diff_calls = 0
    for t in range(test_start + L, test_end - 1):
        if diff_sig_cache is None or (t - (test_start + L)) % refresh == 0:
            scen = predictor.predict(t, num_samples=int(dcfg.get("num_samples", 16)), ddim_steps=int(dcfg.get("ddim_steps", 15)), rel=rel)
            raw = scen["next_day_return_mean"] / (np.abs(scen["path_vol"]) + 1e-6)
            diff_sig_cache = np.nan_to_num(raw)
            n_diff_calls += 1
        mom_z = feats["mom_21"][t]
        rl_signal = None
        if rl_agent is not None and infer_env is not None:
            infer_env.t = t
            infer_env.weights = live_weights.copy()
            infer_env.equity = live_equity[-1]
            infer_env.drawdown = live_dd
            state = infer_env._state()
            with torch.no_grad():
                a, _, _ = rl_agent.act(state, deterministic=True)
            rl_signal = base_signal[t] + resid_scale * np.tanh(np.asarray(a, dtype=np.float64))
            rl_signal = np.clip(rl_signal, -3.0, 3.0) / 3.0
        sig = combine_signals(rl_signal, diff_sig_cache, mom_z, float(scfg.get("w_rl", 0.45)), float(scfg.get("w_diffusion", 0.35)), float(scfg.get("w_momentum", 0.20)))
        signals[t] = sig

        vol_target_daily = float(rcfg.get("vol_target_annual", 0.15)) / np.sqrt(252.0)
        w_now = signals_to_weights(
            sig[None],
            daily_vol[t][None],
            vol_target_daily=vol_target_daily,
            max_weight=float(rcfg.get("max_weight", 0.25)),
            max_leverage=float(rcfg.get("max_leverage", 1.0)),
            min_abs_signal=float(scfg.get("min_abs_signal", 0.05)),
        )[0]
        r_next = returns_daily[t + 1]
        cost = float(rcfg.get("cost_bps", 5.0)) / 1e4
        pr = float((w_now * r_next).sum() - np.abs(w_now - live_weights).sum() * cost)
        live_equity.append(live_equity[-1] * float(np.exp(np.log1p(np.clip(pr, -0.9, 10.0)))))
        live_dd = min(live_dd, 1.0 - live_equity[-1] / max(live_equity))
        live_weights = w_now
        if (t - test_start) % 60 == 0:
            log(f"walkforward day {t - test_start}/{test_end - test_start} | {time.time() - t0:.0f}s | diffusion calls {n_diff_calls}")

    scfg_all = cfg.section("strategy")
    signals = smooth_signal(signals, int(scfg_all.get("signal_ema", 5)))

    vol_target_daily = float(rcfg.get("vol_target_annual", 0.15)) / np.sqrt(252.0)
    weights_alpha = signals_to_weights(
        signals,
        daily_vol,
        vol_target_daily=vol_target_daily * 4.0,
        max_weight=float(rcfg.get("max_weight", 0.25)),
        max_leverage=1.0,
        min_abs_signal=float(scfg.get("min_abs_signal", 0.05)),
    )

    trend_window = int(scfg_all.get("trend_window", 200))
    bull = build_bull_mask(arrays, trend_window)
    core_gross = 1.0
    weights_core = core_trend_weights(
        daily_vol,
        bull,
        vol_target_daily=vol_target_daily,
        max_weight=float(rcfg.get("max_weight", 0.25)),
        core_gross=core_gross,
        bear_multiplier=float(scfg_all.get("bear_multiplier", 0.3)),
    )
    satellite_share = float(scfg_all.get("satellite_share", 0.30))
    weights = combine_core_satellite(weights_core, weights_alpha, satellite_share)
    weights[:test_start] = 0.0
    weights_core[:test_start] = 0.0
    weights_alpha[:test_start] = 0.0

    engine = BacktestEngine(cost_bps=float(bcfg.get("cost_bps", 5.0)))
    net, equity = engine.run(weights, returns_daily)

    mom_only_signals = zscore(feats["mom_21"])
    w_mom = signals_to_weights(mom_only_signals, daily_vol, vol_target_daily=vol_target_daily * 4.0, max_weight=float(rcfg.get("max_weight", 0.25)), max_leverage=1.0)
    w_mom[:test_start] = 0.0
    net_mom, eq_mom = engine.run(w_mom, returns_daily)

    net_c, eq_c = engine.run(weights_core, returns_daily)
    net_a, eq_a = engine.run(weights_alpha, returns_daily)

    bh_w = np.zeros_like(weights)
    bh_w[:, :] = 1.0 / N
    bh_w[:test_start] = 0.0
    net_bh, eq_bh = BacktestEngine(cost_bps=0.0).run(bh_w, returns_daily)

    sl = slice(test_start, test_end)
    reports = [
        full_report("PEDC Alpha (Core+Satellite)", net[sl][:-1], equity[sl], weights[sl]),
        full_report("Core trend-only", net_c[sl][:-1], eq_c[sl], weights_core[sl]),
        full_report("Alpha satellite-only", net_a[sl][:-1], eq_a[sl], weights_alpha[sl]),
        full_report("Buy&Hold equal-weight", net_bh[sl][:-1], eq_bh[sl], bh_w[sl]),
    ]

    return {
        "reports": reports,
        "equity": {"alpha": equity[sl], "core": eq_c[sl], "satellite": eq_a[sl], "momentum": eq_mom[sl], "buyhold": eq_bh[sl]},
        "dates": dates[sl],
        "weights": weights[sl],
        "signals": signals[sl],
        "net_returns": {"alpha": net[sl], "core": net_c[sl], "satellite": net_a[sl], "momentum": net_mom[sl], "buyhold": net_bh[sl]},
        "test_range": (int(test_start), int(test_end)),
        "elapsed_s": time.time() - t0,
    }
