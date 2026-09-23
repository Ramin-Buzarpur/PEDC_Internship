"""
روش ۱: Maximum Likelihood Estimation (MLE) برای حرکت براونی هندسی (GBM)

فرض: لگاریتم بازده‌های روزانه i.i.d. نرمال هستند. mu و sigma با MLE از روی
بازده‌های لگاریتمی تاریخی (تا همان روز مبنا، بدون دیدن آینده) برآورد می‌شوند:

    mu_hat    = میانگین نمونه‌ای بازده‌های لگاریتمی   (MLE برای میانگین نرمال)
    sigma_hat = انحراف‌معیار نمونه‌ای (ddof=0)          (MLE برای واریانس نرمال)

پیش‌بینی h روز جلوتر:  E[log S_h] = log(S0) + h*mu_hat ,  Var = h*sigma_hat^2
کریدور: صدک‌های نرمال با سطح اطمینان مشخص (پیش‌فرض ۹۰٪، z=1.645)
"""
import argparse
from pathlib import Path

import numpy as np
from scipy.stats import norm

from wf_common import run_walkforward, save_dataset, summarize

HERE = Path(__file__).resolve().parent
METHOD_NAME = "mle_gbm"


def fit_mle_gbm(history_slice):
    log_prices = np.log(history_slice['Close'].values)
    log_returns = np.diff(log_prices)
    mu_hat = log_returns.mean()          # MLE برای میانگین
    sigma_hat = log_returns.std(ddof=0)  # MLE برای انحراف معیار
    return mu_hat, sigma_hat


def forecast_fn(history_slice, horizon, confidence_level=0.90):
    mu_hat, sigma_hat = fit_mle_gbm(history_slice)
    s0 = float(history_slice['Close'].iloc[-1])
    z = norm.ppf(0.5 + confidence_level / 2.0)

    predicted = np.zeros(horizon + 1)
    upper = np.zeros(horizon + 1)
    lower = np.zeros(horizon + 1)

    predicted[0] = s0
    upper[0] = s0
    lower[0] = s0

    for h in range(1, horizon + 1):
        mean_log = np.log(s0) + h * mu_hat
        std_log = sigma_hat * np.sqrt(h)
        predicted[h] = np.exp(mean_log)
        upper[h] = np.exp(mean_log + z * std_log)
        lower[h] = np.exp(mean_log - z * std_log)

    return predicted, upper, lower


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="دیتاست walk-forward با MLE GBM")
    parser.add_argument("--ticker", default="BZ=F")
    parser.add_argument("--fetch-period", default="10y")
    parser.add_argument("--num-cutoffs", type=int, default=250, help="تعداد روزهای مبنا از گذشته")
    parser.add_argument("--horizon", type=int, default=7)
    parser.add_argument("--confidence", type=float, default=0.90)
    parser.add_argument("--min-history", type=int, default=250)
    parser.add_argument("--out", default=str(HERE / f"oil_dataset_{METHOD_NAME}.csv"))
    args = parser.parse_args()

    df = run_walkforward(
        method_name=METHOD_NAME,
        forecast_fn=lambda hist, hz: forecast_fn(hist, hz, confidence_level=args.confidence),
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
