"""
روش ۲: برآوردگر انقباضی جیمز-اشتاین (James-Stein) برای درفت بازده

ایدهٔ اصلی (پارادوکس اشتاین): میانگین نمونه‌ای بازدهٔ روزانه یک برآوردگر خیلی
پرنویز است (تقریباً بی‌فایده برای پیش‌بینی جهت بازار). به‌جای استفاده از میانگین
خام یک پنجرهٔ اخیر، تاریخچه را به K بلوکِ غیرهم‌پوشان تقسیم می‌کنیم، میانگین هر
بلوک را حساب می‌کنیم، و همهٔ K میانگین را با فرمول جیمز-اشتاین به‌سمت میانگین کل
(grand mean) منقبض می‌کنیم. این دقیقاً همان روشی است که در مالی با نام
Bayes-Stein شناخته می‌شود (Jorion, 1986) و برای برآورد بازدهٔ مورد انتظار
پرتفوی استفاده می‌شود - هدفش کاهش خطای برآورد (MSE) با قربانی‌کردن کمی بایاس است.

خروجی نهایی: درفتِ منقبض‌شدهٔ آخرین بلوک، به‌همراه sigma از MLE کامل، در قالب
همان کریدور GBM (مثل روش MLE) - تا تفاوت "فقط عوض کردن برآوردگر drift" شفاف دیده شود.
"""
import argparse
from pathlib import Path

import numpy as np
from scipy.stats import norm

from wf_common import run_walkforward, save_dataset, summarize

HERE = Path(__file__).resolve().parent
METHOD_NAME = "james_stein"


def james_stein_shrink(block_means: np.ndarray, common_var: float) -> np.ndarray:
    """
    فرمول کلاسیک جیمز-اشتاین برای K>=3 میانگین با واریانس مشترک شناخته‌شده:
        shrink = 1 - (K-2)*common_var / sum((mu_i - grand_mean)^2)
        mu_JS_i = grand_mean + shrink * (mu_i - grand_mean)   [shrink clipped به [0,1]]
    """
    K = len(block_means)
    grand_mean = block_means.mean()
    if K < 3:
        return np.full(K, grand_mean)
    ss = np.sum((block_means - grand_mean) ** 2)
    if ss < 1e-18:
        return np.full(K, grand_mean)
    shrink = 1.0 - (K - 2) * common_var / ss
    shrink = np.clip(shrink, 0.0, 1.0)
    return grand_mean + shrink * (block_means - grand_mean)


def fit_james_stein_drift(history_slice, block_len: int = 21, max_blocks: int = 24):
    log_prices = np.log(history_slice['Close'].values)
    log_returns = np.diff(log_prices)

    n_full_blocks = len(log_returns) // block_len
    n_full_blocks = min(n_full_blocks, max_blocks)
    if n_full_blocks < 3:
        # تاریخچهٔ کافی برای انقباض نیست - به MLE ساده سقوط می‌کنیم
        mu = log_returns.mean()
        sigma = log_returns.std(ddof=0)
        return mu, sigma

    usable = log_returns[-n_full_blocks * block_len:]
    blocks = usable.reshape(n_full_blocks, block_len)
    block_means = blocks.mean(axis=1)
    block_vars = blocks.var(axis=1, ddof=1)

    common_var = block_vars.mean() / block_len  # واریانس تخمینِ میانگین هر بلوک
    shrunk_means = james_stein_shrink(block_means, common_var)

    mu_shrunk_latest = shrunk_means[-1]
    sigma_hat = log_returns.std(ddof=0)  # sigma از کل تاریخچه (MLE ساده)
    return mu_shrunk_latest, sigma_hat


def forecast_fn(history_slice, horizon, confidence_level=0.90, block_len=21, max_blocks=24):
    mu_shrunk, sigma_hat = fit_james_stein_drift(history_slice, block_len=block_len, max_blocks=max_blocks)
    s0 = float(history_slice['Close'].iloc[-1])
    z = norm.ppf(0.5 + confidence_level / 2.0)

    predicted = np.zeros(horizon + 1)
    upper = np.zeros(horizon + 1)
    lower = np.zeros(horizon + 1)

    predicted[0] = s0
    upper[0] = s0
    lower[0] = s0

    for h in range(1, horizon + 1):
        mean_log = np.log(s0) + h * mu_shrunk
        std_log = sigma_hat * np.sqrt(h)
        predicted[h] = np.exp(mean_log)
        upper[h] = np.exp(mean_log + z * std_log)
        lower[h] = np.exp(mean_log - z * std_log)

    return predicted, upper, lower


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="دیتاست walk-forward با انقباض James-Stein روی درفت")
    parser.add_argument("--ticker", default="BZ=F")
    parser.add_argument("--fetch-period", default="10y")
    parser.add_argument("--num-cutoffs", type=int, default=250)
    parser.add_argument("--horizon", type=int, default=7)
    parser.add_argument("--confidence", type=float, default=0.90)
    parser.add_argument("--min-history", type=int, default=300)
    parser.add_argument("--block-len", type=int, default=21, help="طول هر بلوک (روز) برای میانگین‌گیری")
    parser.add_argument("--max-blocks", type=int, default=24)
    parser.add_argument("--out", default=str(HERE / f"oil_dataset_{METHOD_NAME}.csv"))
    args = parser.parse_args()

    df = run_walkforward(
        method_name=METHOD_NAME,
        forecast_fn=lambda hist, hz: forecast_fn(
            hist, hz, confidence_level=args.confidence,
            block_len=args.block_len, max_blocks=args.max_blocks,
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
