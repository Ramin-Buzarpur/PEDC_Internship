"""
آزمایش کنترل‌شده: تأثیر «گام تصویر پایستگی انرژی» روی پهنای کریدور

فرضیه: خط y_next = y_next * (target_F / F_current) در
_rk4_geodesic_step_with_projection همهٔ مسیرهای شبیه‌سازی‌شده را به یک سطح
انرژی ثابت قفل می‌کند و همین باعث می‌شود کریدور عدم‌قطعیت مصنوعی تنگ بماند -
مستقل از تعداد سناریوها (که قبلاً تست و رد شد) یا تعداد بُعدهای منیفلد.

اینجا test.py دست‌نخورده می‌ماند. یک زیرکلاس ساخته می‌شود که فقط همان یک متد را
بازنویسی می‌کند - دقیقاً همان کد اصلی، منهای بلوک نهاییِ بازمقیاس‌سازی انرژی.
همه‌چیز دیگر (بُعد منیفلد m=3، تعبیهٔ تاکنز، k_scenarios، متریک محلی لدویت-ولف
و ...) کاملاً دست‌نخورده و یکسان با نسخهٔ اصلی می‌ماند - فقط همین یک قید حذف
می‌شود تا اثرش به‌تنهایی دیده شود.
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from build_dataset import FinslerGeodesicSimulationEngine, fetch_full_history, build_manifold_from_slice

HERE = Path(__file__).resolve().parent


class NoEnergyProjectionEngine(FinslerGeodesicSimulationEngine):
    """
    کپیِ دقیقِ _rk4_geodesic_step_with_projection از test.py، فقط با حذفِ
    بلوکِ بازمقیاس‌سازیِ نهاییِ انرژی. هیچ خط دیگری تغییر نکرده است.
    """

    def _rk4_geodesic_step_with_projection(self, x: np.ndarray, y: np.ndarray, dt: float, target_F: float):
        G1, _, _, _ = self._compute_spray_coefficients(x, y)
        k1_x = y
        k1_y = -2.0 * G1

        x_k2 = x + 0.5 * dt * k1_x
        y_k2 = y + 0.5 * dt * k1_y
        G2, _, _, _ = self._compute_spray_coefficients(x_k2, y_k2)
        k2_x = y_k2
        k2_y = -2.0 * G2

        x_k3 = x + 0.5 * dt * k2_x
        y_k3 = y + 0.5 * dt * k2_y
        G3, _, _, _ = self._compute_spray_coefficients(x_k3, y_k3)
        k3_x = y_k3
        k3_y = -2.0 * G3

        x_k4 = x + dt * k3_x
        y_k4 = y + dt * k3_y
        G4, _, _, _ = self._compute_spray_coefficients(x_k4, y_k4)
        k4_x = y_k4
        k4_y = -2.0 * G4

        x_next = x + (dt / 6.0) * (k1_x + 2 * k2_x + 2 * k3_x + k4_x)
        y_next = y + (dt / 6.0) * (k1_y + 2 * k2_y + 2 * k3_y + k4_y)

        # --- قید پایستگی انرژی عمداً حذف شده (تنها تفاوت با نسخهٔ اصلی) ---

        return x_next, y_next


def run_one(engine_cls, raw_slice, ticker, k_neighbors, horizon, k_scenarios, dt):
    engine = engine_cls(ticker=ticker, period="10y", k_neighbors=k_neighbors)
    build_manifold_from_slice(engine, raw_slice)
    engine.simulate_poincare_geodesics(horizon=horizon, k_scenarios=k_scenarios, dt=dt)
    return engine.geodesic_results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="آزمایش ابلیشن قید پایستگی انرژی")
    parser.add_argument("--ticker", default="BZ=F")
    parser.add_argument("--fetch-period", default="10y")
    parser.add_argument("--num-cutoffs", type=int, default=3, help="همان تعداد روز مبنای تست قبلی")
    parser.add_argument("--horizon", type=int, default=7)
    parser.add_argument("--k-scenarios", type=int, default=9, help="۹ - همان مقدار اصلی، چون نشان دادیم تعدادش اثری ندارد")
    parser.add_argument("--k-neighbors", type=int, default=35)
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

        res_control = run_one(FinslerGeodesicSimulationEngine, raw_slice, args.ticker,
                               args.k_neighbors, args.horizon, args.k_scenarios, args.dt)
        res_variant = run_one(NoEnergyProjectionEngine, raw_slice, args.ticker,
                               args.k_neighbors, args.horizon, args.k_scenarios, args.dt)

        for step in range(args.horizon + 1):
            width_control = float(res_control['upper_corridor'][step] - res_control['lower_corridor'][step])
            width_variant = float(res_variant['upper_corridor'][step] - res_variant['lower_corridor'][step])
            rows.append({
                'as_of_date': as_of_date.date().isoformat(),
                'step_ahead': step,
                'corridor_width_with_projection': width_control,
                'corridor_width_without_projection': width_variant,
                'ratio': (width_variant / width_control) if width_control > 0 else np.nan,
            })
            print(f"  step {step}: با قید={width_control:.3f}  |  بدون قید={width_variant:.3f}  "
                  f"|  نسبت={width_variant / width_control if width_control > 0 else float('nan'):.2f}x")

    df = pd.DataFrame(rows)
    out_path = HERE / "experiment_energy_projection_ablation.csv"
    df.to_csv(out_path, index=False, encoding="utf-8-sig")
    print(f"\nذخیره شد: {out_path}")

    print("\n=== خلاصه (میانگین نسبت پهنای کریدور بدون‌قید / با‌قید، به‌ازای گام) ===")
    print(df.groupby('step_ahead')['ratio'].mean().to_string())
