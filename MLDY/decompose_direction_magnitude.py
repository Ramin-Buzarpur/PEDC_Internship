"""
تفکیک ارزیابی به «جهت» و «مقدار جهت» - به‌جای فقط MAE خام یا hit-rate کریدور.

منطق: در بازارهای مالی، پیش‌بینیِ ۹۹٪ دقیق بی‌معناست. چیزی که واقعاً ارزش
معاملاتی دارد این است که مدل چقدر درست «جهت» حرکت (بالا/پایین) را تشخیص
می‌دهد، و از میان پیش‌بینی‌های درست‌جهت، چقدر «اندازهٔ» حرکت را هم درست گرفته.

سه معیار:
  - direction_accuracy : درصد مواقعی که علامت حرکت پیش‌بینی‌شده با علامت حرکت
                         واقعی یکی بوده (بی‌ربط به اینکه اندازه چقدر دقیق بوده)
  - magnitude_corr      : همبستگی پیرسون بین |حرکت پیش‌بینی‌شده| و |حرکت واقعی| -
                         یعنی آیا وقتی مدل «مطمئن‌تر» است (حرکت بزرگ‌تر پیش‌بینی
                         می‌کند) واقعاً حرکت بزرگ‌تری هم رخ می‌دهد؟
  - weighted_score      : w_dir * direction_accuracy + w_mag * (magnitude_corr+1)/2
                         (وزن‌ها قابل‌تنظیم با --w-direction و --w-magnitude)
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent


def load_and_normalize(path: str, predicted_col: str, method_name: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    if predicted_col != 'predicted_price' and predicted_col in df.columns:
        df = df.rename(columns={predicted_col: 'predicted_price'})
    if 'method' not in df.columns:
        df['method'] = method_name
    return df


def decompose(df: pd.DataFrame, w_direction: float = 0.6, w_magnitude: float = 0.4) -> pd.DataFrame:
    d = df[(df['step_ahead'] > 0) & df['actual_price'].notna()].copy()
    d['predicted_move'] = d['predicted_price'] - d['as_of_price']
    d['actual_move'] = d['actual_price'] - d['as_of_price']

    # نویز خیلی‌کوچیک نزدیک صفر را «بدون‌جهت واضح» حساب نمی‌کنیم؛ فقط علامت
    d['direction_correct'] = np.sign(d['predicted_move']) == np.sign(d['actual_move'])

    results = []
    for (method, step), g in d.groupby(['method', 'step_ahead']):
        direction_accuracy = g['direction_correct'].mean()
        if g['predicted_move'].abs().std() > 0 and g['actual_move'].abs().std() > 0 and len(g) > 2:
            magnitude_corr = np.corrcoef(g['predicted_move'].abs(), g['actual_move'].abs())[0, 1]
        else:
            magnitude_corr = np.nan
        magnitude_mae = (g['actual_move'].abs() - g['predicted_move'].abs()).abs().mean()
        mag_component = (magnitude_corr + 1) / 2 if not np.isnan(magnitude_corr) else 0.5
        weighted_score = w_direction * direction_accuracy + w_magnitude * mag_component
        results.append({
            'method': method,
            'step_ahead': step,
            'n': len(g),
            'direction_accuracy': direction_accuracy,
            'magnitude_corr': magnitude_corr,
            'magnitude_mae': magnitude_mae,
            'weighted_score': weighted_score,
        })
    return pd.DataFrame(results).sort_values(['method', 'step_ahead'])


def overall_summary(detail: pd.DataFrame, w_direction: float, w_magnitude: float) -> pd.DataFrame:
    rows = []
    for method, g in detail.groupby('method'):
        avg_dir = (g['direction_accuracy'] * g['n']).sum() / g['n'].sum()
        avg_mag_corr = np.nanmean(g['magnitude_corr'])
        mag_component = (avg_mag_corr + 1) / 2 if not np.isnan(avg_mag_corr) else 0.5
        score = w_direction * avg_dir + w_magnitude * mag_component
        rows.append({
            'method': method,
            'overall_direction_accuracy': avg_dir,
            'overall_magnitude_corr': avg_mag_corr,
            'overall_weighted_score': score,
            'n_total': g['n'].sum(),
        })
    return pd.DataFrame(rows).sort_values('overall_weighted_score', ascending=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="تفکیک دقت جهت و مقدار برای همهٔ روش‌های موجود")
    parser.add_argument("--w-direction", type=float, default=0.6)
    parser.add_argument("--w-magnitude", type=float, default=0.4)
    args = parser.parse_args()

    total_w = args.w_direction + args.w_magnitude
    w_dir, w_mag = args.w_direction / total_w, args.w_magnitude / total_w

    sources = [
        (HERE / "oil_geodesic_dataset.csv", 'frechet_predicted_price', 'finsler_geodesic (k_neighbors=35)'),
        (HERE / "oil_geodesic_dataset_k5.csv", 'frechet_predicted_price', 'finsler_geodesic (k_neighbors=5)'),
        (HERE / "oil_dataset_mle_gbm.csv", 'predicted_price', 'mle_gbm'),
        (HERE / "oil_dataset_james_stein.csv", 'predicted_price', 'james_stein'),
        (HERE / "oil_dataset_mc_variance_reduction.csv", 'predicted_price', 'mc_variance_reduction'),
        (HERE / "oil_dataset_garch11.csv", 'predicted_price', 'garch11'),
        (HERE / "oil_dataset_historical_bootstrap.csv", 'predicted_price', 'historical_bootstrap'),
    ]

    frames = []
    for path, pred_col, name in sources:
        if path.exists():
            frames.append(load_and_normalize(str(path), pred_col, name))
        else:
            print(f"(رد شد - پیدا نشد: {path.name})")

    combined = pd.concat(frames, ignore_index=True)
    detail = decompose(combined, w_dir, w_mag)
    summary = overall_summary(detail, w_dir, w_mag)

    detail.to_csv(HERE / "direction_magnitude_detail.csv", index=False, encoding="utf-8-sig")
    summary.to_csv(HERE / "direction_magnitude_summary.csv", index=False, encoding="utf-8-sig")

    print(f"وزن نهایی: جهت={w_dir:.2f} | مقدار={w_mag:.2f}\n")
    print("=== خلاصهٔ کلی (مرتب بر اساس weighted_score، بهترین اول) ===\n")
    print(summary.to_string(index=False))
    print("\n=== جزئیات به‌ازای گام (step_ahead) ===\n")
    print(detail.to_string(index=False))
