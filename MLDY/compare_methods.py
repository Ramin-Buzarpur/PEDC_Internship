"""
مقایسهٔ همهٔ روش‌های تولید دیتاست (Finsler + ۵ روش آماری) کنار هم.

دیتاست فینسلر (oil_geodesic_dataset.csv) شمای ستونی متفاوتی دارد (ساخته‌شدهٔ
build_dataset.py، فقط ۲ روز مبنا، بدون تضمین برچسب کامل) - اینجا به شمای مشترک
نگاشت می‌شود تا حداقل قابل مقایسهٔ کیفی باشد. برای مقایسهٔ آماریِ واقعاً منصفانه،
دیتاست فینسلر هم باید با همان تعداد روز مبنا و همان انتخاب cutoff رو-به-گذشته
دوباره ساخته شود (فعلاً فقط ۲ نمونه دارد و قابل اتکا نیست).
"""
import glob
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent


def load_common_datasets():
    frames = []
    for path in sorted(glob.glob(str(HERE / "oil_dataset_*.csv"))):
        df = pd.read_csv(path)
        frames.append(df)
    if frames:
        return pd.concat(frames, ignore_index=True)
    return pd.DataFrame()


def load_finsler_if_present():
    path = HERE / "oil_geodesic_dataset.csv"
    if not path.exists():
        return None
    df = pd.read_csv(path)
    df = df.rename(columns={'frechet_predicted_price': 'predicted_price'})
    df['method'] = 'finsler_geodesic'
    df['confidence_level'] = np.nan  # فینسلر یک سطح اطمینان صریح آماری تعریف نمی‌کند
    keep = ['method', 'ticker', 'as_of_date', 'as_of_price', 'step_ahead', 'forecast_date',
            'predicted_price', 'upper_corridor', 'lower_corridor', 'corridor_width',
            'confidence_level', 'actual_price', 'prediction_error', 'abs_pct_error',
            'within_corridor']
    for c in keep:
        if c not in df.columns:
            df[c] = np.nan
    return df[keep]


def summarize_all(df: pd.DataFrame) -> pd.DataFrame:
    realized = df[df['actual_price'].notna() & (df['step_ahead'] > 0)].copy()
    if realized.empty:
        return pd.DataFrame()
    realized['within_corridor'] = realized['within_corridor'].astype(float)
    grp = realized.groupby('method').agg(
        n_forecasts=('prediction_error', 'count'),
        mae=('prediction_error', lambda s: s.abs().mean()),
        rmse=('prediction_error', lambda s: np.sqrt((s ** 2).mean())),
        mape=('abs_pct_error', 'mean'),
        hit_rate_within_corridor=('within_corridor', 'mean'),
        avg_corridor_width=('corridor_width', 'mean'),
    ).reset_index().sort_values('mae')
    return grp


if __name__ == "__main__":
    stat_df = load_common_datasets()
    finsler_df = load_finsler_if_present()

    parts = [d for d in [stat_df, finsler_df] if d is not None and not d.empty]
    if not parts:
        print("هیچ دیتاستی پیدا نشد. اول روش‌ها را اجرا کن.")
        raise SystemExit(1)

    combined = pd.concat(parts, ignore_index=True)
    combined.to_csv(HERE / "all_methods_combined.csv", index=False, encoding="utf-8-sig")

    summary = summarize_all(combined)
    summary.to_csv(HERE / "methods_comparison_summary.csv", index=False, encoding="utf-8-sig")

    print("=== خلاصهٔ مقایسهٔ کل روش‌ها (مرتب بر اساس MAE، بهترین اول) ===\n")
    print(summary.to_string(index=False))

    if finsler_df is not None:
        n_finsler_realized = finsler_df[finsler_df['actual_price'].notna() & (finsler_df['step_ahead'] > 0)].shape[0]
        print(f"\n⚠ توجه: دیتاست finsler_geodesic فقط {n_finsler_realized} ردیف برچسب واقعی دارد "
              f"(فقط ۲ روز مبنا، از لبهٔ امروز) - نتایجش آماری معتبر نیست، فقط برای دید کلی است.")
