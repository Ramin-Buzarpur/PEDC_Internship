"""
دیتاست‌ساز بر پایهٔ FinslerGeodesicSimulationEngine (test.py)

برای هر روز مبنا (as_of_date)، فقط داده‌ی تا همان روز به موتور داده می‌شود (walk-forward
صادقانه، بدون دیدن آینده)، سپس پیش‌بینی افق N روزه (میانگین فرشه + کریدور بالا/پایین)
گرفته می‌شود و با قیمت واقعی که بعداً رخ داده مقایسه می‌شود.

خروجی: یک فایل CSV با یک ردیف به‌ازای هر (as_of_date, step_ahead).
"""
import argparse
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf
from scipy.signal import savgol_filter
from scipy.spatial import cKDTree
from scipy.stats import iqr

HERE = Path(__file__).resolve().parent

# --- بارگذاری کلاس از test.py بدون اجرای بلوک دمویش ---
_spec = importlib.util.spec_from_file_location("finsler_engine_module", HERE / "test.py")
_finsler_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_finsler_mod)
FinslerGeodesicSimulationEngine = _finsler_mod.FinslerGeodesicSimulationEngine


def fetch_full_history(ticker: str, period: str) -> pd.DataFrame:
    """یک‌بار کل تاریخچهٔ قیمت را می‌گیرد؛ بقیهٔ کد از روی این، برش‌های walk-forward می‌سازد."""
    ticker_obj = yf.Ticker(ticker)
    raw = ticker_obj.history(period=period, interval="1d", auto_adjust=False)
    if raw.empty:
        raw = yf.download(ticker, period=period, interval="1d", auto_adjust=False)
        if isinstance(raw.columns, pd.MultiIndex):
            raw.columns = raw.columns.get_level_values(0)
    raw = raw[['Close', 'Volume']].dropna()
    raw = raw[(raw['Close'] > 0) & (raw['Volume'] > 0)].copy()
    raw.index = pd.to_datetime(raw.index).tz_localize(None)
    return raw


def build_manifold_from_slice(engine: FinslerGeodesicSimulationEngine, raw_slice: pd.DataFrame) -> None:
    """
    همان منطق بخش دوم fetch_and_reconstruct (بعد از دانلود) را روی یک برش از دادهٔ
    از قبل‌گرفته‌شده اجرا می‌کند - بدون اینکه دوباره از یاهو فایننس بخواند و بدون
    اینکه هیچ فرمول/الگوریتمی نسبت به نسخهٔ اصلی test.py تغییر کند.
    """
    engine.df = raw_slice
    engine.dates = pd.to_datetime(raw_slice.index)
    engine.prices = raw_slice['Close'].values
    engine.log_p = np.log(engine.prices)

    win = int(np.clip(len(engine.log_p) // 100, 11, 25))
    if win % 2 == 0:
        win += 1
    engine.s_dot = savgol_filter(engine.log_p, window_length=win, polyorder=3, deriv=1)

    data = engine.log_p
    iqr_val = max(iqr(data), 1e-6)
    n_bins = int(np.clip(np.ceil((data.max() - data.min()) / (2.0 * iqr_val * len(data) ** (-1 / 3))), 16, 48))
    bins = np.linspace(data.min(), data.max(), n_bins + 1)
    digitized = np.clip(np.digitize(data, bins) - 1, 0, n_bins - 1)

    ami_scores = []
    for t in range(1, 21):
        x1, x2 = digitized[:-t], digitized[t:]
        p_joint = np.zeros((n_bins, n_bins))
        np.add.at(p_joint, (x1, x2), 1.0)
        p_joint /= len(x1)
        p_x1 = np.bincount(x1, minlength=n_bins) / len(x1)
        p_x2 = np.bincount(x2, minlength=n_bins) / len(x1)
        nz = p_joint > 0
        ami_scores.append(np.sum(p_joint[nz] * np.log2(p_joint[nz] / np.outer(p_x1, p_x2)[nz])))

    derivatives = np.diff(ami_scores)
    local_min = np.where((derivatives[:-1] < 0) & (derivatives[1:] > 0))[0] + 1
    engine.tau = int(local_min[0] + 1) if len(local_min) > 0 else 4

    N = len(engine.log_p)
    tau = engine.tau
    X_cols = [engine.log_p[(2 - j) * tau: N - j * tau if j > 0 else N] for j in range(3)]
    Y_cols = [engine.s_dot[(2 - j) * tau: N - j * tau if j > 0 else N] for j in range(3)]

    engine.X = np.column_stack(X_cols)
    engine.Y = np.column_stack(Y_cols)
    engine.manifold_dates = engine.dates[2 * tau:]
    engine.tree = cKDTree(engine.X)


def build_dataset(
    ticker: str = "BZ=F",
    fetch_period: str = "10y",
    num_cutoffs: int = 2,
    horizon: int = 7,
    k_scenarios: int = 9,
    k_neighbors: int = 35,
    dt: float = 0.20,
    min_history_rows: int = 400,
) -> pd.DataFrame:
    print(f"[۱/۳] دانلود کل تاریخچهٔ {ticker} ({fetch_period})...")
    full_raw = fetch_full_history(ticker, fetch_period)
    print(f"      {len(full_raw)} روز معاملاتی موجود، از {full_raw.index[0].date()} تا {full_raw.index[-1].date()}")

    if len(full_raw) < min_history_rows + num_cutoffs:
        raise ValueError("تاریخچهٔ کافی برای ساخت این تعداد cutoff وجود ندارد.")

    # روزهای مبنا: آخرین num_cutoffs روز معاملاتیِ موجود (قدیمی‌ترین اول)
    cutoff_positions = list(range(len(full_raw) - num_cutoffs, len(full_raw)))

    rows = []
    for c_i, pos in enumerate(cutoff_positions, start=1):
        as_of_date = full_raw.index[pos]
        raw_slice = full_raw.iloc[: pos + 1]  # فقط تا همین روز، آینده دیده نمی‌شود

        print(f"[۲/۳] ({c_i}/{len(cutoff_positions)}) شبیه‌سازی با مبنای {as_of_date.date()} "
              f"({len(raw_slice)} روز دادهٔ ورودی)...")

        engine = FinslerGeodesicSimulationEngine(ticker=ticker, period=fetch_period, k_neighbors=k_neighbors)
        build_manifold_from_slice(engine, raw_slice)
        engine.simulate_poincare_geodesics(horizon=horizon, k_scenarios=k_scenarios, dt=dt)

        res = engine.geodesic_results
        mean_energy = float(np.mean(res['energies']))

        for step, forecast_date in enumerate(res['forecast_dates']):
            forecast_date_norm = pd.Timestamp(forecast_date).normalize()
            # قیمت واقعی را (اگر تا امروز رخ داده باشد) از تاریخچهٔ کامل پیدا می‌کنیم
            match = full_raw.index[full_raw.index.normalize() == forecast_date_norm]
            actual_price = float(full_raw.loc[match[0], 'Close']) if len(match) > 0 else np.nan

            frechet_price = float(res['frechet_price'][step])
            upper = float(res['upper_corridor'][step])
            lower = float(res['lower_corridor'][step])

            rows.append({
                'ticker': ticker,
                'as_of_date': as_of_date.date().isoformat(),
                'as_of_price': float(engine.prices[-1]),
                'tau': engine.tau,
                'step_ahead': step,
                'forecast_date': forecast_date_norm.date().isoformat(),
                'frechet_predicted_price': frechet_price,
                'upper_corridor': upper,
                'lower_corridor': lower,
                'corridor_width': upper - lower,
                'manifold_x1': float(res['frechet_path'][step, 0]),
                'manifold_x2': float(res['frechet_path'][step, 1]),
                'manifold_x3': float(res['frechet_path'][step, 2]),
                'mean_scenario_energy': mean_energy,
                'k_scenarios_used': len(res['energies']),
                'actual_price': actual_price,
                'prediction_error': (actual_price - frechet_price) if not np.isnan(actual_price) else np.nan,
                'abs_pct_error': (abs(actual_price - frechet_price) / actual_price * 100.0)
                if not np.isnan(actual_price) else np.nan,
                'within_corridor': (lower <= actual_price <= upper) if not np.isnan(actual_price) else None,
            })

    print("[۳/۳] تمام.")
    return pd.DataFrame(rows)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ساخت دیتاست walk-forward از FinslerGeodesicSimulationEngine")
    parser.add_argument("--ticker", default="BZ=F")
    parser.add_argument("--fetch-period", default="10y")
    parser.add_argument("--num-cutoffs", type=int, default=2, help="تعداد روزهای مبنا (پیش‌فرض: ۲ روز گذشته)")
    parser.add_argument("--horizon", type=int, default=7)
    parser.add_argument("--k-scenarios", type=int, default=9)
    parser.add_argument("--k-neighbors", type=int, default=35)
    parser.add_argument("--dt", type=float, default=0.20)
    parser.add_argument("--out", default=str(HERE / "oil_geodesic_dataset.csv"))
    args = parser.parse_args()

    df = build_dataset(
        ticker=args.ticker,
        fetch_period=args.fetch_period,
        num_cutoffs=args.num_cutoffs,
        horizon=args.horizon,
        k_scenarios=args.k_scenarios,
        k_neighbors=args.k_neighbors,
        dt=args.dt,
    )
    df.to_csv(args.out, index=False, encoding="utf-8-sig")
    print(f"\nذخیره شد: {args.out}  ({len(df)} ردیف)")
    print(df.to_string(index=False))
