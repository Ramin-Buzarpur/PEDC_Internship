"""
حساسیت به تعداد سال‌های تاریخچه (fetch_period) - آیا یه بازهٔ خاص (کمتر یا
بیشتر از ۷ سال فعلی) کریدور/دقت جهت را واقعاً بهتر می‌کند؟

نکتهٔ مهم: تغییر fetch_period فقط طول کل تاریخچهٔ دانلودشده را عوض می‌کند، اما
چون build_manifold_from_slice همیشه از کل دادهٔ موجود تا لحظهٔ cutoff استفاده
می‌کند (نه یک پنجرهٔ لغزان با طول ثابت)، این آزمایش عملاً «چقدر تاریخچهٔ گذشته
در دسترس مدل باشد» را می‌سنجد، نه یک پنجرهٔ رولینگ با طول ثابت.
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from build_dataset import build_dataset

HERE = Path(__file__).resolve().parent


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="حساسیت نتایج فینسلر به تعداد سال‌های تاریخچه")
    parser.add_argument("--ticker", default="BZ=F")
    parser.add_argument("--periods", nargs="+", default=["3y", "5y", "7y", "10y"])
    parser.add_argument("--num-cutoffs", type=int, default=15)
    parser.add_argument("--horizon", type=int, default=7)
    parser.add_argument("--k-scenarios", type=int, default=9)
    parser.add_argument("--k-neighbors", type=int, default=35)
    parser.add_argument("--dt", type=float, default=0.20)
    args = parser.parse_args()

    all_frames = []
    for period in args.periods:
        print(f"\n########## fetch_period={period} ##########")
        try:
            df = build_dataset(
                ticker=args.ticker,
                fetch_period=period,
                num_cutoffs=args.num_cutoffs,
                horizon=args.horizon,
                k_scenarios=args.k_scenarios,
                k_neighbors=args.k_neighbors,
                dt=args.dt,
            )
        except Exception as exc:  # noqa: BLE001
            print(f"رد شد ({period}): {exc}")
            continue
        df['lookback_period'] = period
        out_path = HERE / f"oil_geodesic_dataset_lookback_{period}.csv"
        df.to_csv(out_path, index=False, encoding="utf-8-sig")
        print(f"ذخیره شد: {out_path}")
        all_frames.append(df)

    if not all_frames:
        raise SystemExit("هیچ بازه‌ای موفق نشد.")

    combined = pd.concat(all_frames, ignore_index=True)
    combined.to_csv(HERE / "lookback_sensitivity_combined.csv", index=False, encoding="utf-8-sig")

    realized = combined[(combined['step_ahead'] > 0) & combined['actual_price'].notna()].copy()
    realized['within_corridor'] = realized['within_corridor'].astype(float)
    realized['predicted_move'] = realized['frechet_predicted_price'] - realized['as_of_price']
    realized['actual_move'] = realized['actual_price'] - realized['as_of_price']
    realized['direction_correct'] = (np.sign(realized['predicted_move']) == np.sign(realized['actual_move'])).astype(float)

    summary = realized.groupby('lookback_period').agg(
        mae=('prediction_error', lambda s: s.abs().mean()),
        hit_rate=('within_corridor', 'mean'),
        direction_accuracy=('direction_correct', 'mean'),
        avg_corridor_width=('corridor_width', 'mean'),
        n=('prediction_error', 'count'),
    ).reset_index().sort_values('mae')

    summary.to_csv(HERE / "lookback_sensitivity_summary.csv", index=False, encoding="utf-8-sig")
    print("\n=== خلاصهٔ حساسیت به تعداد سال‌های تاریخچه ===\n")
    print(summary.to_string(index=False))
