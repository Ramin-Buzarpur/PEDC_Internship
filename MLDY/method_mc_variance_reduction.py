"""
روش ۳: شبیه‌سازی مونت‌کارلوی GBM با کاهش واریانس (Antithetic Variates)

mu/sigma با همان MLE روش ۱ برآورد می‌شوند، اما به‌جای فرمول تحلیلی، مسیرهای
قیمت را با مونت‌کارلو شبیه‌سازی می‌کنیم تا صدک‌های واقعی (نه فقط نرمال) به‌دست
بیاید. برای کاهش واریانس برآوردگر مونت‌کارلو از Antithetic Variates استفاده
می‌شود: به‌جای M نمونهٔ مستقل Z، برای هر Z یک نمونهٔ آینه‌ای -Z هم شبیه‌سازی
می‌شود؛ چون تابع lognormal یکنواخت نیست، این کار واریانس میانگین برآوردی را به‌طور
قابل‌توجهی کم می‌کند بدون افزایش تعداد فراخوانی RNG مستقل.

در انتهای اسکریپت، واریانس با/بدون antithetic گزارش می‌شود تا کاهش واریانس
عملاً اندازه‌گیری شود، نه فقط ادعا شود.
"""
import argparse
from pathlib import Path

import numpy as np

from wf_common import run_walkforward, save_dataset, summarize

HERE = Path(__file__).resolve().parent
METHOD_NAME = "mc_variance_reduction"

_rng = np.random.default_rng(42)  # seed ثابت برای بازتولیدپذیری


def fit_mle_gbm(history_slice):
    log_prices = np.log(history_slice['Close'].values)
    log_returns = np.diff(log_prices)
    return log_returns.mean(), log_returns.std(ddof=0)


def simulate_antithetic(s0, mu_hat, sigma_hat, horizon, n_pairs=2000):
    """برمی‌گرداند: شبیه‌سازی قیمت در هر گام h برای همهٔ 2*n_pairs مسیر."""
    z = _rng.standard_normal((n_pairs, horizon))
    z_full = np.concatenate([z, -z], axis=0)  # antithetic pairing

    log_increments = mu_hat + sigma_hat * z_full  # (2*n_pairs, horizon)
    cum_log = np.cumsum(log_increments, axis=1)
    log_s0 = np.log(s0)
    sim_prices = np.exp(log_s0 + cum_log)  # (2*n_pairs, horizon) -> ستون h = قیمت در گام h+1
    return sim_prices


def forecast_fn(history_slice, horizon, confidence_level=0.90, n_pairs=2000):
    mu_hat, sigma_hat = fit_mle_gbm(history_slice)
    s0 = float(history_slice['Close'].iloc[-1])

    sim_prices = simulate_antithetic(s0, mu_hat, sigma_hat, horizon, n_pairs=n_pairs)

    lower_pct = (1 - confidence_level) / 2.0 * 100
    upper_pct = 100 - lower_pct

    predicted = np.zeros(horizon + 1)
    upper = np.zeros(horizon + 1)
    lower = np.zeros(horizon + 1)

    predicted[0] = s0
    upper[0] = s0
    lower[0] = s0

    for h in range(1, horizon + 1):
        col = sim_prices[:, h - 1]
        predicted[h] = np.median(col)  # median پایدارتر از mean برای توزیع چوله lognormal
        lower[h] = np.percentile(col, lower_pct)
        upper[h] = np.percentile(col, upper_pct)

    return predicted, upper, lower


def report_variance_reduction(history_slice, horizon, n_pairs=2000):
    """مقایسهٔ واریانس برآورد میانگین با/بدون antithetic - فقط برای گزارش تشخیصی."""
    mu_hat, sigma_hat = fit_mle_gbm(history_slice)
    s0 = float(history_slice['Close'].iloc[-1])

    z_plain = _rng.standard_normal((2 * n_pairs, horizon))
    plain_prices = np.exp(np.log(s0) + np.cumsum(mu_hat + sigma_hat * z_plain, axis=1))

    anti_prices = simulate_antithetic(s0, mu_hat, sigma_hat, horizon, n_pairs=n_pairs)

    h_last = horizon - 1
    var_plain = plain_prices[:, h_last].var(ddof=1)
    var_anti = anti_prices[:, h_last].var(ddof=1)
    reduction_pct = (1 - var_anti / var_plain) * 100 if var_plain > 0 else 0.0
    print(f"[{METHOD_NAME}] واریانس MC ساده در گام آخر: {var_plain:.4f} | "
          f"با antithetic: {var_anti:.4f} | کاهش: {reduction_pct:.1f}%")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="دیتاست walk-forward با مونت‌کارلوی antithetic")
    parser.add_argument("--ticker", default="BZ=F")
    parser.add_argument("--fetch-period", default="10y")
    parser.add_argument("--num-cutoffs", type=int, default=250)
    parser.add_argument("--horizon", type=int, default=7)
    parser.add_argument("--confidence", type=float, default=0.90)
    parser.add_argument("--min-history", type=int, default=250)
    parser.add_argument("--n-pairs", type=int, default=2000, help="تعداد جفت‌های antithetic (کل مسیرها = ۲×این عدد)")
    parser.add_argument("--out", default=str(HERE / f"oil_dataset_{METHOD_NAME}.csv"))
    args = parser.parse_args()

    # یک نمونهٔ تشخیصی از کاهش واریانس، روی آخرین برش تاریخچه
    diag_full = None
    try:
        from wf_common import fetch_full_history
        diag_full = fetch_full_history(args.ticker, args.fetch_period)
        report_variance_reduction(diag_full, args.horizon, n_pairs=args.n_pairs)
    except Exception as exc:  # noqa: BLE001
        print(f"[{METHOD_NAME}] گزارش تشخیصی واریانس رد شد: {exc}")

    df = run_walkforward(
        method_name=METHOD_NAME,
        forecast_fn=lambda hist, hz: forecast_fn(
            hist, hz, confidence_level=args.confidence, n_pairs=args.n_pairs,
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
