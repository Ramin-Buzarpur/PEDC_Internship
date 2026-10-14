# -*- coding: utf-8 -*-
"""
compare_diffusion_data.py
==========================
Answers: does training the diffusion scenario generator on the full
~23-year Dukascopy EUR/USD history produce better 3-day-ahead scenarios
than training it on the old ~2.8-year Yahoo hourly window (the original
train_diffusion_eurusd.py setup)?

Both models are trained with the *same* context_len / horizon_len /
n_timesteps, on data that ends strictly before a shared cutoff, and are
then walk-forward tested on the same held-out origins past that cutoff —
so neither model has seen the test period, and architecture/config
differences cannot explain the result.

Metrics per model, over all walk-forward origins:
  - MAE / RMSE of the scenario-ensemble mean's 3-day cumulative log-return
    vs. the realized one
  - Directional accuracy (up/down call)
  - Pinball (quantile) loss averaged over 10/25/50/75/90% scenario
    quantiles -- a CRPS-style proxy for "are the generated scenarios well
    calibrated", not just "is the mean close"
  - Naive persistence baseline (predict zero return) for context

Usage:
    python compare_diffusion_data.py            # train both + backtest
    python compare_diffusion_data.py --skip-train   # reuse existing checkpoints
"""

from __future__ import annotations

import os
import sys
import time

import numpy as np
import pandas as pd

import train_diffusion_eurusd as diff_mod

DATA_DIR = diff_mod.DATA_DIR
FULL_CSV = os.path.join(DATA_DIR, "EUR_USD_hourly_full.csv")
OLD_CSV = diff_mod.HOURLY_CSV  # original ~2.8y Yahoo hourly csv

CKPT_OLD = os.path.join(DATA_DIR, "diffusion_old_window.pt")
CKPT_NEW = os.path.join(DATA_DIR, "diffusion_full_history.pt")

CONTEXT_LEN = 168
HORIZON_MULTIPLIER = 3
N_TIMESTEPS = 100
EPOCHS = 10
N_SCENARIOS = 60
N_TEST_ORIGINS = 30
STEP_BACK = 24  # hours between consecutive walk-forward origins
QUANTILE_LEVELS = [10, 25, 50, 75, 90]


def pinball_loss(actual: float, quantile_preds: dict[int, float]) -> float:
    losses = []
    for q, pred in quantile_preds.items():
        tau = q / 100.0
        diff = actual - pred
        losses.append(max(tau * diff, (tau - 1) * diff))
    return float(np.mean(losses))


def main() -> None:
    skip_train = "--skip-train" in sys.argv

    print("Loading full-history hourly log returns ...")
    full_log_ret, bars_per_day = diff_mod.load_hourly_log_returns(FULL_CSV)
    horizon_len = int(round(bars_per_day * HORIZON_MULTIPLIER))
    print(f"full history bars: {len(full_log_ret)}  bars/day~={bars_per_day:.1f}  horizon_len={horizon_len}")

    old_log_ret_full, _ = diff_mod.load_hourly_log_returns(OLD_CSV)
    old_window_bars = len(old_log_ret_full)
    print(f"old-dataset window length to match: {old_window_bars} bars (~2.8y)")

    test_reserve = N_TEST_ORIGINS * STEP_BACK + horizon_len + CONTEXT_LEN + 10
    cutoff = len(full_log_ret) - test_reserve
    if cutoff - old_window_bars < CONTEXT_LEN + horizon_len:
        raise ValueError("Not enough full history before cutoff to carve out an old-sized window.")

    new_train_log_ret = full_log_ret[:cutoff]
    old_train_log_ret = full_log_ret[cutoff - old_window_bars : cutoff]
    print(f"cutoff index: {cutoff}  new-model train bars: {len(new_train_log_ret)}  "
          f"old-model train bars: {len(old_train_log_ret)}")

    if not skip_train:
        for label, log_ret, ckpt in [
            ("OLD-window (~2.8y)", old_train_log_ret, CKPT_OLD),
            ("FULL-history (~23y)", new_train_log_ret, CKPT_NEW),
        ]:
            print(f"\n=== Training diffusion model on {label} data -> {ckpt} ===")
            t0 = time.time()
            diff_mod.train(
                context_len=CONTEXT_LEN,
                horizon_multiplier=HORIZON_MULTIPLIER,
                n_timesteps=N_TIMESTEPS,
                epochs=EPOCHS,
                checkpoint_path=ckpt,
                log_ret_override=log_ret,
                bars_per_day_override=bars_per_day,
            )
            print(f"  done in {time.time() - t0:.1f}s")
    else:
        print("Skipping training; reusing existing checkpoints.")

    # ------------------------------------------------------------------
    # Walk-forward backtest on held-out origins, both models vs naive.
    # ------------------------------------------------------------------
    origins = list(range(cutoff, cutoff + N_TEST_ORIGINS * STEP_BACK, STEP_BACK))
    origins = [o for o in origins if o + horizon_len <= len(full_log_ret) and o - CONTEXT_LEN >= 0]
    print(f"\nWalk-forward origins: {len(origins)}")

    rows = []
    for oi, origin in enumerate(origins):
        ctx = full_log_ret[origin - CONTEXT_LEN : origin]
        actual_future = full_log_ret[origin : origin + horizon_len]
        actual_cum = float(actual_future.sum())

        row = {"origin_idx": origin, "actual_cum_return": actual_cum}
        for label, ckpt in [("old", CKPT_OLD), ("new", CKPT_NEW)]:
            samples, _ = diff_mod.sample_scenarios(
                n_scenarios=N_SCENARIOS, checkpoint_path=ckpt, recent_log_returns=ctx
            )
            cum_per_scenario = samples.sum(axis=1)
            row[f"{label}_mean"] = float(cum_per_scenario.mean())
            row[f"{label}_std"] = float(cum_per_scenario.std())
            qpreds = {q: float(np.percentile(cum_per_scenario, q)) for q in QUANTILE_LEVELS}
            row[f"{label}_pinball"] = pinball_loss(actual_cum, qpreds)
            row[f"{label}_q50"] = qpreds[50]

        rows.append(row)
        print(f"  [{oi + 1}/{len(origins)}] origin={origin} actual={actual_cum:+.5f} "
              f"old_mean={row['old_mean']:+.5f} new_mean={row['new_mean']:+.5f}")

    df = pd.DataFrame(rows)
    out_csv = os.path.join(DATA_DIR, "compare_diffusion_data_results.csv")
    df.to_csv(out_csv, index=False)
    print(f"\nSaved per-origin results: {out_csv}")

    # ------------------------------------------------------------------
    # Scoring
    # ------------------------------------------------------------------
    results = []
    for label in ["old", "new"]:
        err = df[f"{label}_mean"] - df["actual_cum_return"]
        mae = err.abs().mean()
        rmse = np.sqrt((err**2).mean())
        pred_up = df[f"{label}_mean"] > 0
        actual_up = df["actual_cum_return"] > 0
        dir_acc = (pred_up == actual_up).mean()
        avg_pinball = df[f"{label}_pinball"].mean()
        avg_spread = df[f"{label}_std"].mean()
        results.append(
            {"model": label, "MAE": mae, "RMSE": rmse, "directional_accuracy": dir_acc,
             "avg_pinball_CRPS_proxy": avg_pinball, "avg_scenario_std": avg_spread}
        )

    # naive persistence baseline (predict zero return)
    naive_err = -df["actual_cum_return"]
    results.append(
        {
            "model": "naive_persistence",
            "MAE": naive_err.abs().mean(),
            "RMSE": np.sqrt((naive_err**2).mean()),
            "directional_accuracy": (df["actual_cum_return"] <= 0).mean(),
            "avg_pinball_CRPS_proxy": df["actual_cum_return"].abs().mean() / 2,
            "avg_scenario_std": 0.0,
        }
    )

    score_df = pd.DataFrame(results).sort_values("MAE")
    pd.set_option("display.float_format", lambda x: f"{x:.6f}")
    print("\n=== Comparison: diffusion trained on OLD (~2.8y) vs FULL (~23y) EUR/USD history ===")
    print(score_df.to_string(index=False))

    old_row = score_df[score_df["model"] == "old"].iloc[0]
    new_row = score_df[score_df["model"] == "new"].iloc[0]
    better_mae = "new (full-history)" if new_row["MAE"] < old_row["MAE"] else "old (2.8y)"
    better_pinball = "new (full-history)" if new_row["avg_pinball_CRPS_proxy"] < old_row["avg_pinball_CRPS_proxy"] else "old (2.8y)"
    print(f"\n=> Lower point-forecast error (MAE): {better_mae}")
    print(f"=> Better-calibrated scenarios (pinball/CRPS proxy): {better_pinball}")


if __name__ == "__main__":
    main()
