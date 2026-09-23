"""
آزمایش کنترل‌شده: تأثیر k_neighbors (میانگین‌گیری محلی) روی پهنای کریدور

فرضیه: در _get_local_metrics، دینامیک محلی از میانگین‌گیری روی k_neighbors نقطهٔ
همسایه به‌دست می‌آید. میانگین‌گیری واریانس را کم می‌کند - پس هرچه k_neighbors
بزرگ‌تر باشد، میدان برداری هموارتر و مسیرهای شبیه‌سازی‌شده کم‌نوسان‌تر می‌شوند.
اینجا هیچ خطی از test.py تغییر نمی‌کند - فقط همان پارامتر ورودیِ از قبل موجودِ
سازندهٔ کلاس (k_neighbors) با دو مقدار متفاوت فراخوانی می‌شود.
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from build_dataset import FinslerGeodesicSimulationEngine, fetch_full_history, build_manifold_from_slice

HERE = Path(__file__).resolve().parent


def run_one(raw_slice, ticker, k_neighbors, horizon, k_scenarios, dt):
    engine = FinslerGeodesicSimulationEngine(ticker=ticker, period="10y", k_neighbors=k_neighbors)
    build_manifold_from_slice(engine, raw_slice)
    engine.simulate_poincare_geodesics(horizon=horizon, k_scenarios=k_scenarios, dt=dt)
    return engine.geodesic_results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="آزمایش ابلیشن k_neighbors (میانگین‌گیری محلی)")
    parser.add_argument("--ticker", default="BZ=F")
    parser.add_argument("--fetch-period", default="10y")
    parser.add_argument("--num-cutoffs", type=int, default=3)
    parser.add_argument("--horizon", type=int, default=7)
    parser.add_argument("--k-scenarios", type=int, default=9)
    parser.add_argument("--k-neighbors-control", type=int, default=35, help="مقدار اصلی در test.py")
    parser.add_argument("--k-neighbors-variant", type=int, default=5, help="خیلی کمتر = میانگین‌گیری خیلی کمتر")
    parser.add_argument("--dt", type=float, default=0.20)
    args = parser.parse_args()

    print("دانلود تاریخچه...")
    full_raw = fetch_full_history(args.ticker, args.fetch_period)
    cutoff_positions = list(range(len(full_raw) - args.num_cutoffs, len(full_raw)))

    rows = []
    for pos in cutoff_positions:
        as_of_date = full_raw.index[pos]
        raw_slice = full_raw.iloc[: pos + 1]
        print(f"\n=== as_of={as_of_date.date()} ===")

        res_control = run_one(raw_slice, args.ticker, args.k_neighbors_control,
                               args.horizon, args.k_scenarios, args.dt)
        res_variant = run_one(raw_slice, args.ticker, args.k_neighbors_variant,
                               args.horizon, args.k_scenarios, args.dt)

        for step in range(args.horizon + 1):
            width_control = float(res_control['upper_corridor'][step] - res_control['lower_corridor'][step])
            width_variant = float(res_variant['upper_corridor'][step] - res_variant['lower_corridor'][step])
            rows.append({
                'as_of_date': as_of_date.date().isoformat(),
                'step_ahead': step,
                f'corridor_width_k{args.k_neighbors_control}': width_control,
                f'corridor_width_k{args.k_neighbors_variant}': width_variant,
                'ratio': (width_variant / width_control) if width_control > 0 else np.nan,
            })
            print(f"  step {step}: k={args.k_neighbors_control} -> {width_control:.3f}  |  "
                  f"k={args.k_neighbors_variant} -> {width_variant:.3f}  |  "
                  f"نسبت={width_variant / width_control if width_control > 0 else float('nan'):.2f}x")

    df = pd.DataFrame(rows)
    out_path = HERE / "experiment_k_neighbors_ablation.csv"
    df.to_csv(out_path, index=False, encoding="utf-8-sig")
    print(f"\nذخیره شد: {out_path}")

    print(f"\n=== خلاصه (میانگین نسبت پهنای کریدور k={args.k_neighbors_variant} / k={args.k_neighbors_control}, به‌ازای گام) ===")
    print(df.groupby('step_ahead')['ratio'].mean().to_string())
