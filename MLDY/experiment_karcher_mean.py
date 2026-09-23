"""
آزمایش کنترل‌شده: میانگین کارشر (تکراری، گرادیانی) به‌جای میانگین فرشهٔ فعلی
(جست‌وجوی Nelder-Mead روی همون تابع هدف).

هر دو، از نظر ریاضی، دارن همون کمیت رو محاسبه می‌کنن: نقطه‌ای که مجموع مربع
فاصله‌های راندرز تا k_scenarios نقطهٔ مسیر را کمینه می‌کند. تفاوت فقط در روش
عددیِ رسیدن به آن نقطه‌ست. اینجا یک نسخهٔ تکراریِ سبک (شبیه الگوریتم Weiszfeld
تعمیم‌یافته، با استفاده از همون متریک محلیِ لدویت-ولف که در بقیهٔ کد هست) پیاده
می‌شود و پهنای کریدور حاصل با نسخهٔ Nelder-Mead اصلی مقایسه می‌شود.

هیچ خطی از test.py تغییر نمی‌کند - فقط از توابع/متدهای موجودش (_get_local_metrics)
دوباره استفاده می‌شود.
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from build_dataset import FinslerGeodesicSimulationEngine, fetch_full_history, build_manifold_from_slice

HERE = Path(__file__).resolve().parent


def karcher_mean(engine, pts_t: np.ndarray, max_iter: int = 25, tol: float = 1e-7):
    """
    نسخهٔ تکراری/گرادیانیِ میانگین کارشر: هر تکرار، متریک محلی را در نقطهٔ فعلی
    برآورد می‌کند و مرکز را به‌سمت میانگین وزن‌دارشده (وزن = 1/فاصلهٔ راندرز،
    مشابه الگوریتم Weiszfeld) جابه‌جا می‌کند - یعنی به‌جای جست‌وجوی کور
    Nelder-Mead، از ساختار هندسیِ موجود مدل استفاده می‌شود.
    """
    p = pts_t.mean(axis=0)
    for _ in range(max_iter):
        a_mat, _, b_vec, _ = engine._get_local_metrics(p)
        deltas = pts_t - p
        alpha = np.sqrt(np.clip(np.einsum('ij,jk,ik->i', deltas, a_mat, deltas), 1e-12, None))
        beta = deltas @ b_vec
        F = np.clip(alpha + beta, 1e-6, None)
        weights = 1.0 / F
        new_p = (weights[:, None] * pts_t).sum(axis=0) / weights.sum()
        if np.linalg.norm(new_p - p) < tol:
            p = new_p
            break
        p = new_p
    return p


def frechet_mean_nelder_mead(engine, pts_t: np.ndarray):
    """دقیقاً همون منطق test.py (خط ۲۸۲-۲۹۴)."""
    from scipy.optimize import minimize

    def frechet_objective(p):
        a_mat, _, b_vec, _ = engine._get_local_metrics(p)
        total = 0.0
        for q in pts_t:
            delta = q - p
            alpha_d = np.sqrt(np.maximum(delta @ a_mat @ delta, 0.0))
            beta_d = np.dot(b_vec, delta)
            total += (alpha_d + beta_d) ** 2
        return total / len(pts_t)

    init_guess = pts_t.mean(axis=0)
    res = minimize(frechet_objective, init_guess, method='Nelder-Mead', options={'maxiter': 80})
    return res.x if res.success else init_guess


def corridor_from_center(engine, pts_t: np.ndarray, p_center: np.ndarray):
    a_opt, _, b_opt, _ = engine._get_local_metrics(p_center)
    sqrt_a11 = np.sqrt(max(a_opt[0, 0], 1e-6))
    b1 = b_opt[0]
    delta_log_prices = pts_t[:, 0] - p_center[0]
    pos = delta_log_prices[delta_log_prices >= 0]
    neg = delta_log_prices[delta_log_prices < 0]
    total_disp = np.std(delta_log_prices) if len(delta_log_prices) > 1 else 0.015
    asym_up = np.clip(1.0 / (1.0 + (b1 / sqrt_a11)), 0.6, 1.8)
    asym_down = np.clip(1.0 / (1.0 - (b1 / sqrt_a11)), 0.6, 1.8)
    sigma_up = np.sqrt(np.mean(pos ** 2)) if len(pos) > 0 else total_disp
    sigma_down = np.sqrt(np.mean(neg ** 2)) if len(neg) > 0 else total_disp
    std_up = np.clip(sigma_up * asym_up, 0.012, 0.16)
    std_down = np.clip(sigma_down * asym_down, 0.012, 0.16)
    upper = np.exp(p_center[0] + std_up)
    lower = np.exp(p_center[0] - std_down)
    return upper, lower


def run_engine(raw_slice, ticker, k_neighbors, horizon, k_scenarios, dt):
    engine = FinslerGeodesicSimulationEngine(ticker=ticker, period="10y", k_neighbors=k_neighbors)
    build_manifold_from_slice(engine, raw_slice)
    engine.simulate_poincare_geodesics(horizon=horizon, k_scenarios=k_scenarios, dt=dt)
    return engine


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="آزمایش میانگین کارشر در برابر میانگین فرشهٔ Nelder-Mead")
    parser.add_argument("--ticker", default="BZ=F")
    parser.add_argument("--fetch-period", default="10y")
    parser.add_argument("--num-cutoffs", type=int, default=3)
    parser.add_argument("--horizon", type=int, default=7)
    parser.add_argument("--k-scenarios", type=int, default=9)
    parser.add_argument("--k-neighbors", type=int, default=35, help="همون مقدار اصلی - فقط روش میانگین‌گیری تغییر می‌کند")
    parser.add_argument("--dt", type=float, default=0.20)
    args = parser.parse_args()

    print("دانلود تاریخچه...")
    full_raw = fetch_full_history(args.ticker, args.fetch_period)
    cutoff_positions = list(range(len(full_raw) - args.num_cutoffs, len(full_raw)))

    rows = []
    for pos in cutoff_positions:
        as_of_date = full_raw.index[pos]
        raw_slice = full_raw.iloc[: pos + 1]
        print(f"\n=== as_of={as_of_date.date()} ===")

        engine = run_engine(raw_slice, args.ticker, args.k_neighbors, args.horizon, args.k_scenarios, args.dt)
        trajs = engine.geodesic_results['trajectories']

        for step in range(args.horizon + 1):
            pts_t = trajs[:, step, :]

            p_nm = frechet_mean_nelder_mead(engine, pts_t)
            upper_nm, lower_nm = corridor_from_center(engine, pts_t, p_nm)
            width_nm = upper_nm - lower_nm

            p_karcher = karcher_mean(engine, pts_t)
            upper_k, lower_k = corridor_from_center(engine, pts_t, p_karcher)
            width_k = upper_k - lower_k

            center_shift = float(abs(p_karcher[0] - p_nm[0]))

            rows.append({
                'as_of_date': as_of_date.date().isoformat(),
                'step_ahead': step,
                'corridor_width_nelder_mead': float(width_nm),
                'corridor_width_karcher': float(width_k),
                'ratio': float(width_k / width_nm) if width_nm > 0 else np.nan,
                'center_shift_log_price': center_shift,
            })
            print(f"  step {step}: Nelder-Mead={width_nm:.3f}  |  Karcher={width_k:.3f}  |  "
                  f"نسبت={width_k / width_nm if width_nm > 0 else float('nan'):.3f}x  |  "
                  f"جابه‌جایی مرکز={center_shift:.5f}")

    df = pd.DataFrame(rows)
    out_path = HERE / "experiment_karcher_mean_ablation.csv"
    df.to_csv(out_path, index=False, encoding="utf-8-sig")
    print(f"\nذخیره شد: {out_path}")

    print("\n=== خلاصه (میانگین نسبت پهنای کریدور کارشر/Nelder-Mead، به‌ازای گام) ===")
    print(df.groupby('step_ahead')['ratio'].mean().to_string())
    print("\n=== میانگین جابه‌جایی مرکز (فضای لگاریتم قیمت) ===")
    print(df.groupby('step_ahead')['center_shift_log_price'].mean().to_string())
