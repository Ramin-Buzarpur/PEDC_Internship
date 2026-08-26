from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import load_config
from src.pipeline import build_feature_set, get_device, nan_guard, prepare_data
from src.training.train_diffusion import train_diffusion_model
from src.training.train_rl import train_rl_agent
from src.utils.runtime import get_logger, set_seed


def main():
    ap = argparse.ArgumentParser(description="Train PEDC Alpha (diffusion + RL)")
    ap.add_argument("--config", default=None)
    ap.add_argument("--epochs", type=int, default=None, help="override diffusion epochs")
    ap.add_argument("--updates", type=int, default=None, help="override PPO updates")
    ap.add_argument("--stocks", type=int, default=None, help="override max_stocks")
    ap.add_argument("--only", choices=["all", "diffusion", "rl"], default="all")
    ap.add_argument("--no-news", action="store_true")
    args = ap.parse_args()

    cfg = load_config(args.config)
    if args.epochs:
        cfg.raw["training"]["epochs_diffusion"] = args.epochs
    if args.updates:
        cfg.raw["rl"]["total_updates"] = args.updates
    if args.stocks:
        cfg.raw["data"]["max_stocks"] = args.stocks

    set_seed(int(cfg.get("project.seed", 42)))
    out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "outputs")
    log = get_logger("train", os.path.normpath(out_dir))
    device = get_device(cfg)
    log.info(f"device: {device}")

    dates, symbols, arrays, sentiment = prepare_data(cfg, fetch_news=not args.no_news, log=log.info)
    arrays = nan_guard(arrays)
    feats = build_feature_set(dates, symbols, arrays, sentiment, cfg)

    diff_art = None
    rl_art = None
    if args.only in ("all", "diffusion"):
        diff_art = train_diffusion_model(cfg, arrays, sentiment, device, log=log.info)
    if args.only in ("all", "rl"):
        rl_art = train_rl_agent(cfg, feats, arrays, sentiment, device, log=log.info)

    summary = {
        "symbols": symbols,
        "days": int(arrays["close_adj"].shape[0]),
        "date_range": [str(dates[0].date()), str(dates[-1].date())],
    }
    import json

    if diff_art:
        summary["diffusion_val_loss"] = float(diff_art["best_val_loss"])
        summary["test_range"] = list(map(int, diff_art["splits"]["test"]))
    if rl_art:
        summary["ppo_final_equity"] = [float(x) for x in rl_art["history"][-10:]]
    with open(os.path.join(out_dir, "training_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    log.info(f"training complete: {summary['days']} days x {len(symbols)} stocks")


if __name__ == "__main__":
    main()
