"""
روش ۵ (انتخاب خودم): Historical Simulation / Block Bootstrap (ناپارامتری)

برخلاف روش‌های ۱-۴ که همه فرض می‌کنند بازده‌ها نرمال (یا حداقل با یک مدل
پارامتری خاص) هستند، این روش هیچ فرضی روی شکل توزیع نمی‌گذارد - دقیقاً همان
رویکردی که در مدیریت ریسک برای محاسبهٔ VaR به‌کار می‌رود ("Historical VaR").

از تاریخچهٔ بازده‌های لگاریتمی تا روز مبنا، بلوک‌های پیوستهٔ h-روزه (نه تک‌روزها،
تا خودهمبستگی/نوسان‌خوشه‌ای تا حدی حفظ شود) به‌صورت تصادفی با جایگذاری resample
می‌شوند تا هزاران سناریوی h-روزهٔ ممکن ساخته شود؛ سپس صدک‌های تجربی مستقیماً از
همین سناریوها گرفته می‌شوند - بدون هیچ فرمول نرمال یا لوگ‌نرمال.
"""
import argparse
from pathlib import Path

import numpy as np

from wf_common import run_walkforward, save_dataset, summarize

HERE = Path(__file__).resolve().parent
METHOD_NAME = "historical_bootstrap"

_rng = np.random.default_rng(7)


def block_bootstrap_forecast(history_slice, horizon, n_samples=5000, lookback=750):
    log_prices = np.log(history_slice['Close'].values)
    log_returns = np.diff(log_prices)
    log_returns = log_returns[-lookback:] if len(log_returns) > lookback else log_returns

    n = len(log_returns)
    if n <= horizon:
        # تاریخچهٔ کافی برای بلوک h-روزه نیست؛ با بازگشت‌گذاری تک‌روزه جبران می‌کنیم
        block_starts = _rng.integers(0, n, size=n_samples)
        cum_returns = np.array([
            np.sum(_rng.choice(log_returns, size=horizon, replace=True))
            for _ in range(n_samples)
        ])
        return cum_returns

    max_start = n - horizon
    block_starts = _rng.integers(0, max_start, size=n_samples)
    cum_returns = np.array([log_returns[s: s + horizon].sum() for s in block_starts])
    return cum_returns


def forecast_fn(history_slice, horizon, confidence_level=0.90, n_samples=5000, lookback=750):
    s0 = float(history_slice['Close'].iloc[-1])
    lower_pct = (1 - confidence_level) / 2.0 * 100
    upper_pct = 100 - lower_pct

    predicted = np.zeros(horizon + 1)
    upper = np.zeros(horizon + 1)
    lower = np.zeros(horizon + 1)
    predicted[0] = s0
    upper[0] = s0
    lower[0] = s0

    log_prices = np.log(history_slice['Close'].values)
    log_returns = np.diff(log_prices)
    log_returns = log_returns[-lookback:] if len(log_returns) > lookback else log_returns
    n = len(log_returns)

    for h in range(1, horizon + 1):
        if n > h:
            max_start = n - h
            starts = _rng.integers(0, max_start, size=n_samples)
            cum = np.array([log_returns[s: s + h].sum() for s in starts])
        else:
            cum = np.array([
                _rng.choice(log_returns, size=h, replace=True).sum()
                for _ in range(n_samples)
            ])
        sim_prices = s0 * np.exp(cum)
        predicted[h] = np.median(sim_prices)
        lower[h] = np.percentile(sim_prices, lower_pct)
        upper[h] = np.percentile(sim_prices, upper_pct)

    return predicted, upper, lower


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="دیتاست walk-forward با Historical Bootstrap ناپارامتری")
    parser.add_argument("--ticker", default="BZ=F")
    parser.add_argument("--fetch-period", default="10y")
    parser.add_argument("--num-cutoffs", type=int, default=250)
    parser.add_argument("--horizon", type=int, default=7)
    parser.add_argument("--confidence", type=float, default=0.90)
    parser.add_argument("--min-history", type=int, default=250)
    parser.add_argument("--n-samples", type=int, default=5000)
    parser.add_argument("--lookback", type=int, default=750, help="حداکثر تعداد روز اخیر برای resample (~۳ سال)")
    parser.add_argument("--out", default=str(HERE / f"oil_dataset_{METHOD_NAME}.csv"))
    args = parser.parse_args()

    df = run_walkforward(
        method_name=METHOD_NAME,
        forecast_fn=lambda hist, hz: forecast_fn(
            hist, hz, confidence_level=args.confidence,
            n_samples=args.n_samples, lookback=args.lookback,
        ),
        ticker=args.ticker,
        fetch_period=args.fetch_period,
        num_cutoffs=args.num_cutoffs,
        horizon=args.horizon,
        min_history=args.min_history,
        confidence_level=args.confidence,
    )
    save_dataset(df, args.out)
    print("\nخلاصهٔ عملکرد به‌ازای گام پیش‌بینی:")
    print(summarize(df).to_string(index=False))
