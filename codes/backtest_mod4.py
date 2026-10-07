"""
backtest_mod4.py
================
Walk-forward backtest of Module 4's Finsler-Randers geodesic forecast.

Compares, for a 3-day-ahead horizon, the three existing summary methods
(Frechet mean, Karcher mean, manifold geometric median) against a naive
persistence (random-walk) baseline, using:
  - MAE on price
  - Directional accuracy (did it call up/down correctly)

This answers the question "is picking a different mean actually better?"
with numbers instead of theory. Run directly: python backtest_mod4.py
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from mod4 import FinslerRandersGeodesicEngine, MARKET_CATEGORIES


def run_backtest(
    ticker: str,
    asset_name: str,
    years: int = 5,
    n_scenarios: int = 7,
    horizon_days: int = 3,
    n_folds: int = 200,
    step_back: int = 5,
) -> pd.DataFrame:
    """
    Fetches full history once, then repeatedly truncates it to simulate
    "what the engine would have forecast on day t", and compares the T+3
    forecast against the actual realized price.

    n_folds: how many historical origin points to test.
    step_back: spacing (in trading days) between consecutive origin points.
    """
    base_engine = FinslerRandersGeodesicEngine(
        ticker=ticker, asset_name=asset_name, years=years, n_scenarios=n_scenarios
    )
    base_engine.fetch_and_reconstruct_bundle()
    full_prices = base_engine.prices
    full_dates = base_engine.dates
    n_total = len(full_prices)

    min_history = 200  # must stay above the engine's own 150-bar minimum with margin
    latest_usable_origin = n_total - horizon_days - 1
    earliest_origin = min_history

    if latest_usable_origin <= earliest_origin:
        raise ValueError("Not enough history to backtest with this configuration.")

    origins = list(range(latest_usable_origin, earliest_origin, -step_back))[:n_folds]
    origins.sort()

    rows = []
    for origin in origins:
        train_prices = full_prices[: origin + 1]
        actual_future_price = full_prices[origin + horizon_days]
        current_price = full_prices[origin]
        actual_direction_up = actual_future_price > current_price

        try:
            eng = FinslerRandersGeodesicEngine(
                ticker=ticker, asset_name=asset_name, years=years, n_scenarios=n_scenarios
            )
            # Reuse fetched data instead of re-downloading per fold.
            eng.raw_df = base_engine.raw_df.iloc[: origin + 1]
            eng.dates = full_dates[: origin + 1]
            eng.prices = train_prices
            eng.log_returns = np.diff(np.log(train_prices))
            eng.tau = base_engine.tau
            eng.m = base_engine.m

            n_pts = len(eng.log_returns) - (eng.m - 1) * eng.tau
            if n_pts < eng.k_neighbors + 5:
                continue

            eng.point_cloud = np.column_stack(
                [eng.log_returns[i * eng.tau : i * eng.tau + n_pts] for i in range(eng.m)]
            )
            v_list = []
            for i in range(eng.m):
                s_col = eng.point_cloud[:, i]
                v_col = np.zeros_like(s_col)
                v_col[1:] = s_col[1:] - s_col[:-1]
                v_list.append(v_col)
            eng.tangent_velocities = np.column_stack(v_list)
            from scipy.spatial import cKDTree

            eng.tree = cKDTree(eng.point_cloud)

            eng.simulate_geodesics_and_median = None  # not used
            eng.simulate_geodesics_and_means(horizon_days=horizon_days)

            frechet_pred = eng.frechet_mean_price[-1]
            karcher_pred = eng.karcher_mean_price[-1]
            median_pred = eng.manifold_median_price[-1]
            naive_pred = current_price  # random-walk / persistence baseline

            rows.append(
                {
                    "origin_date": full_dates[origin],
                    "current_price": current_price,
                    "actual_future_price": actual_future_price,
                    "actual_up": actual_direction_up,
                    "frechet_pred": frechet_pred,
                    "karcher_pred": karcher_pred,
                    "median_pred": median_pred,
                    "naive_pred": naive_pred,
                }
            )
        except Exception as exc:  # keep going; report failures at the end
            print(f"  [skip] origin {full_dates[origin].date()}: {exc}")
            continue

    return pd.DataFrame(rows)


def score(df: pd.DataFrame) -> pd.DataFrame:
    methods = ["frechet_pred", "karcher_pred", "median_pred", "naive_pred"]
    results = []
    for m in methods:
        err = df[m] - df["actual_future_price"]
        mae = err.abs().mean()
        rmse = np.sqrt((err**2).mean())
        pred_up = df[m] > df["current_price"]
        directional_acc = (pred_up == df["actual_up"]).mean()
        results.append(
            {
                "method": m.replace("_pred", ""),
                "MAE": mae,
                "RMSE": rmse,
                "directional_accuracy": directional_acc,
                "n_folds": len(df),
            }
        )
    return pd.DataFrame(results).sort_values("MAE")


if __name__ == "__main__":
    import sys

    asset_name = sys.argv[1] if len(sys.argv) > 1 else "EUR/USD"
    ticker = MARKET_CATEGORIES["Forex Majors"].get(asset_name, "EURUSD=X")

    print(f"Running walk-forward backtest for {asset_name} ({ticker})...")
    df = run_backtest(ticker=ticker, asset_name=asset_name, years=5, n_scenarios=7, n_folds=20, step_back=5)

    if df.empty:
        print("No valid folds produced — try lowering min_history or increasing years.")
        sys.exit(1)

    print(f"\nCompleted {len(df)} folds from {df['origin_date'].min().date()} to {df['origin_date'].max().date()}.\n")
    result = score(df)
    pd.set_option("display.float_format", lambda x: f"{x:.5f}")
    print(result.to_string(index=False))

    naive_mae = result.loc[result["method"] == "naive", "MAE"].values[0]
    best = result.iloc[0]
    print(f"\nBest method by MAE: {best['method']} (MAE={best['MAE']:.5f})")
    if best["method"] == "naive":
        print("=> None of Frechet/Karcher/Median beat the naive persistence baseline on this window.")
    else:
        improvement = (naive_mae - best["MAE"]) / naive_mae * 100
        print(f"=> {best['method']} beat naive persistence by {improvement:.1f}% MAE.")
