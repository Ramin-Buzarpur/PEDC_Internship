"""
روش ۴ (انتخاب خودم): GARCH(1,1) برای نوسان شرطی + گام تصادفی بدون درفت

پیشینهٔ روش: در ادبیات مالی، میانگین بازدهٔ روزانه عملاً غیرقابل‌پیش‌بینی است
(نزدیک به گام تصادفی خالص)، اما واریانسِ بازده به‌شدت خوشه‌ای‌ست (روزهای پرنوسان
پشت‌سرهم می‌آیند). به‌جای فرض sigma ثابت (مثل روش MLE)، اینجا با GARCH(1,1)
(برازش‌شده با MLE روی بازده‌های لگاریتمی تا روز مبنا) نوسان شرطی را مدل می‌کنیم و
با فرمول بازگشتی GARCH آن را h روز به جلو پیش‌بینی می‌کنیم - این باعث می‌شود
کریدور در دوره‌های پرنوسان بازار خودش را باز/بسته کند، برخلاف کریدور با پهنای
تقریباً ثابت در روش MLE.

درفت را عمداً صفر می‌گذاریم (فرض گام تصادفی برای میانگین) - چون به لحاظ آماری
اثبات‌شده تلاش برای پیش‌بینی جهت بازده روزانه با میانگین نمونه‌ای تقریباً هیچ
قدرت پیش‌بینی ندارد؛ ارزش این روش صرفاً در نوسانِ شرطیِ واقع‌بینانه‌تر است.
"""
import argparse
import warnings
from pathlib import Path

import numpy as np
from scipy.stats import norm

from wf_common import run_walkforward, save_dataset, summarize

HERE = Path(__file__).resolve().parent
METHOD_NAME = "garch11"

warnings.filterwarnings("ignore", category=UserWarning)


def fit_garch_and_forecast_variance(history_slice, horizon):
    from arch import arch_model

    log_prices = np.log(history_slice['Close'].values)
    log_returns = np.diff(log_prices) * 100.0  # مقیاس درصدی برای پایداری بهینه‌سازی arch

    am = arch_model(log_returns, mean="Zero", vol="Garch", p=1, q=1, dist="normal")
    res = am.fit(disp="off", show_warning=False)

    fc = res.forecast(horizon=horizon, reindex=False)
    daily_var_pct2 = fc.variance.values[-1]  # واریانس روزانه به مقیاس درصد^۲
    daily_var = daily_var_pct2 / (100.0 ** 2)  # بازگشت به مقیاس بازدهٔ لگاریتمی خام

    cumulative_var = np.cumsum(daily_var)  # Var(sum of h independent daily log-returns)
    return cumulative_var


def forecast_fn(history_slice, horizon, confidence_level=0.90):
    s0 = float(history_slice['Close'].iloc[-1])
    z = norm.ppf(0.5 + confidence_level / 2.0)

    predicted = np.zeros(horizon + 1)
    upper = np.zeros(horizon + 1)
    lower = np.zeros(horizon + 1)

    predicted[0] = s0
    upper[0] = s0
    lower[0] = s0

    try:
        cumulative_var = fit_garch_and_forecast_variance(history_slice, horizon)
    except Exception:
        # اگر GARCH روی یک برش خاص همگرا نشد، به sigma ثابت (MLE) سقوط می‌کنیم
        log_returns = np.diff(np.log(history_slice['Close'].values))
        sigma_hat = log_returns.std(ddof=0)
        cumulative_var = np.array([(h + 1) * sigma_hat ** 2 for h in range(horizon)])

    log_s0 = np.log(s0)
    for h in range(1, horizon + 1):
        std_log = np.sqrt(cumulative_var[h - 1])
        predicted[h] = s0  # درفت صفر: بهترین حدس نقطه‌ای، خودِ قیمت روز مبناست
        upper[h] = np.exp(log_s0 + z * std_log)
        lower[h] = np.exp(log_s0 - z * std_log)

    return predicted, upper, lower


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="دیتاست walk-forward با GARCH(1,1) + گام تصادفی")
    parser.add_argument("--ticker", default="BZ=F")
    parser.add_argument("--fetch-period", default="10y")
    parser.add_argument("--num-cutoffs", type=int, default=120, help="کمتر از بقیه چون فیت GARCH کندتره")
    parser.add_argument("--horizon", type=int, default=7)
    parser.add_argument("--confidence", type=float, default=0.90)
    parser.add_argument("--min-history", type=int, default=500)
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
