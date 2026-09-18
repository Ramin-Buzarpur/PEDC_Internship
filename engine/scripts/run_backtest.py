from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.backtest.walkforward import run_walkforward
from src.config import load_config
from src.models.diffusion import ConditionalDiffusion
from src.models.matchs import MaTCHS
from src.models.rl_agent import PPOAgent
from src.pipeline import build_feature_set, get_device, nan_guard, prepare_data
from src.training.train_rl import build_state_feats
from src.utils.runtime import get_logger, set_seed


def main():
    ap = argparse.ArgumentParser(description="Walk-forward backtest of PEDC Alpha")
    ap.add_argument("--config", default=None)
    ap.add_argument("--no-news", action="store_true")
    args = ap.parse_args()

    cfg = load_config(args.config)
    set_seed(int(cfg.get("project.seed", 42)))
    out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "outputs")
    out_dir = os.path.normpath(out_dir)
    log = get_logger("backtest", out_dir)
    device = get_device(cfg)

    dates, symbols, arrays, sentiment = prepare_data(cfg, fetch_news=not args.no_news, log=log.info)
    arrays = nan_guard(arrays)
    feats = build_feature_set(dates, symbols, arrays, sentiment, cfg)

    scfg = cfg.section("dataset")
    mcfg = cfg.section("model")
    dcfg = cfg.section("diffusion")
    gcfg = cfg.section("graph")
    L, H = int(scfg.get("seq_len_past", 30)), int(scfg.get("seq_len_future", 10))

    from src.data.graph import build_relation_matrix
    from src.data.dataset import build_splits

    splits = build_splits(arrays["close_adj"].shape[0], float(cfg.section("training").get("val_ratio", 0.15)), 0.15, int(cfg.section("training").get("purge_gap", 40)))
    rel_np = build_relation_matrix(arrays["close_adj"], 0, splits["train"][1] + L + H, int(gcfg.get("corr_window", 250)), float(gcfg.get("corr_threshold", 0.5)))
    rel = torch.from_numpy(rel_np).to(device)

    C = 6 if sentiment is not None else 5
    model_path = os.path.join(out_dir, "models", "matchs_diffusion.pt")
    ckpt = torch.load(model_path, map_location=device)
    if isinstance(ckpt, dict) and "state_dict" in ckpt:
        C = int(ckpt.get("num_channels", C))
        model = MaTCHS(
            num_channels=C,
            seq_len_total=int(ckpt.get("seq_len_total", L + H)),
            d_channels=int(mcfg.get("d_channels", 48)),
            mrt_layers=int(mcfg.get("mrt_layers", 2)),
            heads_masked=min(int(mcfg.get("heads_masked", 2)), rel_np.shape[-1]),
            heads_free=int(mcfg.get("heads_free", 4)),
            num_groups=rel_np.shape[-1],
            dropout=float(mcfg.get("dropout", 0.1)),
        ).to(device)
        model.load_state_dict(ckpt["state_dict"])
    else:
        model = MaTCHS(
            num_channels=C,
            seq_len_total=L + H,
            d_channels=int(mcfg.get("d_channels", 48)),
            mrt_layers=int(mcfg.get("mrt_layers", 2)),
            heads_masked=min(int(mcfg.get("heads_masked", 2)), rel_np.shape[-1]),
            heads_free=int(mcfg.get("heads_free", 4)),
            num_groups=rel_np.shape[-1],
            dropout=float(mcfg.get("dropout", 0.1)),
        ).to(device)
        model.load_state_dict(ckpt)
    log.info(f"loaded diffusion model from {model_path} (C={C})")
    diffusion = ConditionalDiffusion(
        model,
        num_timesteps=int(dcfg.get("num_timesteps", 60)),
        beta_start=float(dcfg.get("beta_start", 1e-4)),
        beta_end=float(dcfg.get("beta_end", 0.2)),
        seq_len_future=H,
        device=device,
    )

    rl_agent = None
    rl_path = os.path.join(out_dir, "models", "ppo_actor.pt")
    if os.path.exists(rl_path):
        rcfg = cfg.section("rl")
        state_dim, n_actions = PPOAgent.dims_from_path(rl_path)
        rl_agent = PPOAgent(state_dim, n_actions, rcfg, device=device)
        rl_agent.load(rl_path)
        log.info(f"loaded RL actor from {rl_path} (state_dim={state_dim}, actions={n_actions})")
    else:
        log.info("no RL actor found; using heuristic ensemble only")

    results = run_walkforward(cfg, dates, symbols, arrays, sentiment, feats, {"diffusion": diffusion, "rel_matrix": rel}, rl_agent=rl_agent, log=log.info)

    plots_dir = os.path.join(out_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates

    fig, axes = plt.subplots(2, 1, figsize=(14, 10), sharex=True, gridspec_kw={"height_ratios": [3, 1]})
    ax = axes[0]
    d = results["dates"]
    sr_map = {r["name"]: r["sharpe"] for r in results["reports"]}
    lines = [
        ("PEDC Alpha (Core+Satellite)", "alpha", "#d62728"),
        ("Core trend-only", "core", "#2ca02c"),
        ("Alpha satellite-only", "satellite", "#ff7f0e"),
        ("Buy&Hold equal-weight", "buyhold", "#7f7f7f"),
    ]
    for name, key, color in lines:
        eq = results["equity"][key]
        ax.plot(d, eq, label=f"{name}  (SR {sr_map.get(name, 0):.2f})", color=color, linewidth=2 if key == "alpha" else 1.5)
    ax.set_title("PEDC Alpha — Out-of-Sample Walk-Forward Equity Curves (test period, start=1.0)")
    ax.set_ylabel("Growth of $1")
    ax.legend()
    ax.grid(alpha=0.3)
    ax2 = axes[1]
    dd = results["equity"]["alpha"] / np.maximum.accumulate(results["equity"]["alpha"]) - 1
    ax2.fill_between(d, dd * 100, 0, color="#d62728", alpha=0.4)
    ax2.set_ylabel("Drawdown %")
    ax2.grid(alpha=0.3)
    ax2.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    plt.tight_layout()
    plot_path = os.path.join(plots_dir, "backtest_equity.png")
    plt.savefig(plot_path, dpi=150)
    plt.close()

    report_lines = ["# PEDC Alpha — Walk-Forward Backtest Report", "", f"Test period: **{results['dates'][0].date()} → {results['dates'][-1].date()}** ({len(results['dates'])} trading days)", ""]
    report_lines.append("| Strategy | CAGR | Sharpe | Sortino | MaxDD | Calmar | WinRate | Turnover/day | Breakeven bps |")
    report_lines.append("|---|---|---|---|---|---|---|---|---|")
    for r in results["reports"]:
        report_lines.append(
            f"| {r['name']} | {r['cagr']:.1%} | {r['sharpe']:.2f} | {r['sortino']:.2f} | {r['max_drawdown']:.1%} | {r['calmar']:.2f} | {r['win_rate_daily']:.1%} | {r.get('avg_turnover_daily', 0):.3f} | {r.get('breakeven_cost_bps', float('nan')):.1f} |"
        )
    report_text = "\n".join(report_lines) + "\n"
    with open(os.path.join(out_dir, "backtest_report.md"), "w", encoding="utf-8") as f:
        f.write(report_text)
    with open(os.path.join(out_dir, "backtest_metrics.json"), "w") as f:
        json.dump(results["reports"], f, indent=2, default=str)

    print("\n" + report_text)
    log.info(f"plot saved to {plot_path}")


if __name__ == "__main__":
    main()
