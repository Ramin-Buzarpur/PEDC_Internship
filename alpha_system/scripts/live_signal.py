from __future__ import annotations

import argparse
import csv
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.backtest.engine import signals_to_weights
from src.backtest.metrics import zscore
from src.config import load_config
from src.data.dataset import build_splits
from src.data.graph import build_relation_matrix
from src.inference.predictor import ScenarioPredictor
from src.inference.strategy import combine_signals
from src.models.diffusion import ConditionalDiffusion
from src.models.matchs import MaTCHS
from src.models.rl_agent import PPOAgent, PortfolioEnv
from src.pipeline import build_feature_set, get_device, nan_guard, prepare_data
from src.training.train_rl import build_state_feats
from src.utils.runtime import get_logger, set_seed


def main():
    ap = argparse.ArgumentParser(description="Generate live trading signals for the next session")
    ap.add_argument("--config", default=None)
    ap.add_argument("--capital", type=float, default=10000.0, help="capital in USD")
    ap.add_argument("--top", type=int, default=10)
    ap.add_argument("--no-news", action="store_true")
    args = ap.parse_args()

    cfg = load_config(args.config)
    set_seed(int(cfg.get("project.seed", 42)))
    out_dir = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "outputs"))
    log = get_logger("live", out_dir)
    device = get_device(cfg)

    dates, symbols, arrays, sentiment = prepare_data(cfg, fetch_news=not args.no_news, log=log.info)
    arrays = nan_guard(arrays)
    feats = build_feature_set(dates, symbols, arrays, sentiment, cfg)

    scfg = cfg.section("dataset")
    mcfg = cfg.section("model")
    dcfg = cfg.section("diffusion")
    rcfg = cfg.section("rl")
    gcfg = cfg.section("graph")
    L, H = int(scfg.get("seq_len_past", 30)), int(scfg.get("seq_len_future", 10))
    N = len(symbols)
    t_last = int(arrays["close_adj"].shape[0]) - 1

    from src.data.graph import build_relation_matrix as _brm

    splits = build_splits(arrays["close_adj"].shape[0], 0.15, 0.15, int(cfg.section("training").get("purge_gap", 40)))
    rel_np = build_relation_matrix(arrays["close_adj"], max(0, t_last - int(gcfg.get("corr_window", 250))), t_last + 1, int(gcfg.get("corr_window", 250)), float(gcfg.get("corr_threshold", 0.5)))
    rel = torch.from_numpy(rel_np).to(device)

    model_path = os.path.join(out_dir, "models", "matchs_diffusion.pt")
    ckpt = torch.load(model_path, map_location=device)
    C = int(ckpt.get("num_channels", 6 if sentiment is not None else 5))
    T_total = int(ckpt.get("seq_len_total", L + H))
    model = MaTCHS(
        num_channels=C,
        seq_len_total=T_total,
        d_channels=int(mcfg.get("d_channels", 48)),
        mrt_layers=int(mcfg.get("mrt_layers", 2)),
        heads_masked=min(int(mcfg.get("heads_masked", 2)), rel_np.shape[-1]),
        heads_free=int(mcfg.get("heads_free", 4)),
        num_groups=rel_np.shape[-1],
        dropout=float(mcfg.get("dropout", 0.1)),
    ).to(device)
    model.load_state_dict(ckpt["state_dict"])
    diffusion = ConditionalDiffusion(model, num_timesteps=int(ckpt.get("num_timesteps", dcfg.get("num_timesteps", 60))), beta_start=float(dcfg.get("beta_start", 1e-4)), beta_end=float(dcfg.get("beta_end", 0.2)), seq_len_future=H, device=device)

    predictor = ScenarioPredictor(diffusion, arrays, sentiment, L, H, device)
    scen = predictor.predict(t_last, num_samples=int(dcfg.get("num_samples", 32)), ddim_steps=int(dcfg.get("ddim_steps", 15)), rel=rel)
    diff_sig = np.nan_to_num(scen["next_day_return_mean"] / (np.abs(scen["path_vol"]) + 1e-6))
    log.info(f"diffusion scenarios ready: {scen['samples'].shape}")

    rl_signal = None
    rl_path = os.path.join(out_dir, "models", "ppo_actor.pt")
    if os.path.exists(rl_path):
        from src.backtest.metrics import zscore

        state_feats_full = build_state_feats(feats, sentiment)
        daily_vol_all = np.maximum(feats["ewma_vol"] / np.sqrt(252.0), 1e-5)
        returns_daily = (arrays["close_adj"][1:] / arrays["close_adj"][:-1] - 1.0).astype(np.float32)
        returns_daily = np.vstack([np.zeros((1, N), dtype=np.float32), returns_daily])
        base_signal = np.clip(zscore(feats["mom_21"]), -3.0, 3.0) / 3.0
        resid_scale = float(rcfg.get("resid_scale", 1.0))
        env = PortfolioEnv(
            returns_daily, state_feats_full, daily_vol_all,
            start=max(0, t_last - 1), end=t_last + 1,
            vol_target_annual=float(rcfg.get("vol_target_annual", 0.15)),
            max_leverage=float(rcfg.get("max_leverage", 1.0)),
            max_weight=float(rcfg.get("max_weight", 0.25)),
            cost_bps=float(rcfg.get("cost_bps", 5.0)),
            drawdown_penalty=float(rcfg.get("drawdown_penalty", 2.0)),
            base_signal=base_signal,
            resid_scale=resid_scale,
        )
        agent = PPOAgent(env.state_dim, N, rcfg, device=device)
        saved_dim, saved_actions = PPOAgent.dims_from_path(rl_path)
        if saved_dim != env.state_dim or saved_actions != N:
            log.warning(f"PPO checkpoint dims {saved_dim}/{saved_actions} != current {env.state_dim}/{N}; skipping RL component")
            rl_signal = None
            rl_path = None
        else:
            agent.load(rl_path)
            env.t = t_last
            env.weights = np.zeros(N, dtype=np.float32)
            env.drawdown = 0.0
            state = env._state()
            with torch.no_grad():
                a, _, _ = agent.act(state, deterministic=True)
            rl_signal = np.clip(base_signal[t_last] + resid_scale * np.tanh(np.asarray(a, dtype=np.float64)), -3.0, 3.0) / 3.0
            log.info("RL actor loaded (residual over momentum)")

    mom_z = feats["mom_21"][t_last]
    sig = combine_signals(
        rl_signal, diff_sig, mom_z,
        float(cfg.section("strategy").get("w_rl", 0.45)),
        float(cfg.section("strategy").get("w_diffusion", 0.35)),
        float(cfg.section("strategy").get("w_momentum", 0.20)),
    )
    daily_vol = np.maximum(feats["ewma_vol"] / np.sqrt(252.0), 1e-5)[t_last]
    weights = signals_to_weights(
        sig[None], daily_vol[None],
        vol_target_daily=float(rcfg.get("vol_target_annual", 0.15)) / np.sqrt(252.0),
        max_weight=float(rcfg.get("max_weight", 0.25)),
        max_leverage=float(rcfg.get("max_leverage", 1.0)),
        min_abs_signal=float(cfg.section("strategy").get("min_abs_signal", 0.05)),
    )[0]

    order = np.argsort(-weights)
    longs = [i for i in order if weights[i] > 1e-4][: args.top]
    shorts = [i for i in order[::-1] if weights[i] < -1e-4][: args.top]
    last_prices = arrays["close_adj"][t_last]

    print("\n" + "=" * 78)
    print(f"PEDC ALPHA — LIVE SIGNALS for next session ({str(dates[t_last].date())} close)")
    print("=" * 78)
    print(f"{'SYMBOL':<8}{'SIDE':<6}{'WEIGHT%':>9}{'USD':>12}{'E[R]+1d%':>10}{'σ_pred%':>9}{'SENTIMENT':>11}")
    for i in longs + shorts:
        side = "LONG" if weights[i] > 0 else "SHORT"
        sent = sentiment[t_last, i] if sentiment is not None else 0.0
        print(
            f"{symbols[i]:<8}{side:<6}{weights[i] * 100:>8.2f}%{args.capital * abs(weights[i]):>12,.0f}"
            f"{scen['next_day_return_mean'][i] * 100:>10.2f}{abs(scen['path_vol'][i]) * 100:>9.2f}{sent:>11.2f}"
        )
    gross = float(np.abs(weights).sum())
    cash = 1.0 - gross if gross <= 1.0 else 0.0
    print(f"\nGross exposure: {gross:.1%} | Cash buffer: {cash:.1%} | Universe: {N} stocks")
    print("DISCLAIMER: probabilistic research tool — NOT financial advice. Markets carry risk of loss.")

    out_csv = os.path.join(out_dir, f"signals_{dates[t_last].strftime('%Y%m%d')}.csv")
    with open(out_csv, "w", newline="") as f:
        wcsv = csv.writer(f)
        wcsv.writerow(["symbol", "weight", "side", "usd_allocation", "pred_next_day_return", "pred_vol"])
        for i in range(N):
            if abs(weights[i]) > 1e-6:
                wcsv.writerow([symbols[i], float(weights[i]), "LONG" if weights[i] > 0 else "SHORT", float(args.capital * abs(weights[i])), float(scen['next_day_return_mean'][i]), float(abs(scen['path_vol'][i]))])
    print(f"saved: {out_csv}")


if __name__ == "__main__":
    main()
