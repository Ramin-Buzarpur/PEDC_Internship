"""
زیرساخت مشترک walk-forward برای همهٔ روش‌های پیش‌بینی قیمت نفت.

هر اسکریپت روش (mle_gbm.py, james_stein.py, ...) این ماژول را import می‌کند تا:
  - یک‌بار کل تاریخچهٔ قیمت را بگیرد
  - روزهای مبنا (as_of_date) را طوری انتخاب کند که برای همهٔ گام‌های افق، قیمت
    واقعی از قبل موجود باشد (بر خلاف نسخهٔ اول build_dataset.py که تا لبهٔ امروز
    می‌رفت و اکثر برچسب‌ها NaN می‌شدند)
  - ردیف‌های خروجی را با یک schema ثابت بسازد تا دیتاست‌های همهٔ روش‌ها قابل مقایسهٔ
    مستقیم باشند.
"""
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf

HERE = Path(__file__).resolve().parent

# ستون‌های مشترک خروجی همهٔ روش‌ها - برای مقایسهٔ apple-to-apple
SCHEMA_COLUMNS = [
    'method', 'ticker', 'as_of_date', 'as_of_price', 'step_ahead', 'forecast_date',
    'predicted_price', 'upper_corridor', 'lower_corridor', 'corridor_width',
    'confidence_level', 'actual_price', 'prediction_error', 'abs_pct_error',
    'within_corridor',
]


def fetch_full_history(ticker: str = "BZ=F", period: str = "10y") -> pd.DataFrame:
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


def make_backward_cutoff_positions(n_total: int, num_cutoffs: int, horizon: int, min_history: int) -> list:
    """
    روزهای مبنا را از گذشته انتخاب می‌کند (نه از لبهٔ امروز) به‌طوری که برای هر
    کدام، `horizon` روز معاملاتیِ آینده از قبل در دیتاست موجود باشد - یعنی همهٔ
    ردیف‌های خروجی برچسب واقعی (actual_price) دارند، نه NaN.
    """
    last_valid_pos = n_total - 1 - horizon
    start_pos = last_valid_pos - num_cutoffs + 1
    if start_pos < min_history:
        raise ValueError(
            f"تاریخچهٔ کافی نیست: با {n_total} روز داده، حداکثر "
            f"{last_valid_pos - min_history + 1} روز مبنا با افق {horizon} و "
            f"حداقل {min_history} روز تاریخچهٔ اولیه ممکن است."
        )
    return list(range(start_pos, last_valid_pos + 1))


def lookup_actual_price(full_raw: pd.DataFrame, forecast_date: pd.Timestamp):
    fd = pd.Timestamp(forecast_date).normalize()
    match = full_raw.index[full_raw.index.normalize() == fd]
    return float(full_raw.loc[match[0], 'Close']) if len(match) > 0 else np.nan


def build_row(method, ticker, as_of_date, as_of_price, step, forecast_date,
              predicted_price, upper, lower, confidence_level, actual_price):
    err = (actual_price - predicted_price) if not np.isnan(actual_price) else np.nan
    abs_pct = (abs(err) / actual_price * 100.0) if not np.isnan(actual_price) and actual_price != 0 else np.nan
    within = (lower <= actual_price <= upper) if not np.isnan(actual_price) else None
    return {
        'method': method,
        'ticker': ticker,
        'as_of_date': as_of_date.date().isoformat() if hasattr(as_of_date, 'date') else str(as_of_date),
        'as_of_price': float(as_of_price),
        'step_ahead': step,
        'forecast_date': pd.Timestamp(forecast_date).date().isoformat(),
        'predicted_price': float(predicted_price),
        'upper_corridor': float(upper),
        'lower_corridor': float(lower),
        'corridor_width': float(upper - lower),
        'confidence_level': confidence_level,
        'actual_price': actual_price,
        'prediction_error': err,
        'abs_pct_error': abs_pct,
        'within_corridor': within,
    }


def run_walkforward(method_name: str, forecast_fn, ticker: str, fetch_period: str,
                     num_cutoffs: int, horizon: int, min_history: int, confidence_level: float,
                     extra_kwargs: dict | None = None):
    """
    اسکلت مشترک: برای هر cutoff گذشته، فقط دادهٔ تا همان روز را به forecast_fn
    می‌دهد. forecast_fn باید (predicted_prices[h+1], upper[h+1], lower[h+1]) را
    برگرداند که ایندکس ۰ = خود as_of_date (بدون عدم قطعیت) و ایندکس ۱..h
    گام‌های آینده باشند.
    """
    extra_kwargs = extra_kwargs or {}
    print(f"[{method_name}] دانلود تاریخچهٔ {ticker} ({fetch_period})...")
    full_raw = fetch_full_history(ticker, fetch_period)
    print(f"[{method_name}] {len(full_raw)} روز معاملاتی، از {full_raw.index[0].date()} تا {full_raw.index[-1].date()}")

    positions = make_backward_cutoff_positions(len(full_raw), num_cutoffs, horizon, min_history)

    rows = []
    for i, pos in enumerate(positions, start=1):
        as_of_date = full_raw.index[pos]
        history_slice = full_raw.iloc[: pos + 1]
        spot = float(history_slice['Close'].iloc[-1])

        predicted, upper, lower = forecast_fn(history_slice, horizon, **extra_kwargs)

        forecast_dates = pd.bdate_range(start=as_of_date, periods=horizon + 1)
        for step in range(horizon + 1):
            fdate = forecast_dates[step]
            actual = lookup_actual_price(full_raw, fdate)
            rows.append(build_row(
                method_name, ticker, as_of_date, spot, step, fdate,
                predicted[step], upper[step], lower[step], confidence_level, actual,
            ))

        if i % max(1, len(positions) // 10) == 0 or i == len(positions):
            print(f"[{method_name}] {i}/{len(positions)} روز مبنا پردازش شد "
                  f"(آخرین: {as_of_date.date()})")

    df = pd.DataFrame(rows, columns=SCHEMA_COLUMNS)
    return df


def save_dataset(df: pd.DataFrame, out_path: str):
    df.to_csv(out_path, index=False, encoding="utf-8-sig")
    print(f"ذخیره شد: {out_path}  ({len(df)} ردیف)")


def summarize(df: pd.DataFrame) -> pd.DataFrame:
    """خلاصهٔ عملکرد به‌ازای step_ahead - برای دیدن سریع کیفیت هر روش."""
    realized = df[df['actual_price'].notna() & (df['step_ahead'] > 0)].copy()
    if realized.empty:
        return pd.DataFrame()
    # نکته: در pandas 3.x مشاهده شد که groupby().agg('mean') روی ستون بولیِ
    # dtype=object (ترکیب True/False/None) مقدار غلط برمی‌گرداند - تبدیل صریح
    # به float این باگ را دور می‌زند.
    realized['within_corridor'] = realized['within_corridor'].astype(float)
    grp = realized.groupby('step_ahead').agg(
        mae=('prediction_error', lambda s: s.abs().mean()),
        rmse=('prediction_error', lambda s: np.sqrt((s ** 2).mean())),
        mape=('abs_pct_error', 'mean'),
        hit_rate_within_corridor=('within_corridor', 'mean'),
        avg_corridor_width=('corridor_width', 'mean'),
        n=('prediction_error', 'count'),
    ).reset_index()
    return grp
