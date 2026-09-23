import numpy as np
import pandas as pd
import yfinance as yf
from scipy.signal import savgol_filter
from scipy.spatial import cKDTree
from scipy.stats import iqr
from scipy.optimize import minimize
from sklearn.covariance import ledoit_wolf
from datetime import datetime, timedelta
import plotly.graph_objects as go


class FinslerGeodesicSimulationEngine:
    """
    ماژول ۴ (نسخه اصلاح‌شده و پایدار):
    شبیه‌ساز ژئودزیک‌های قطعی و میانگین فرشه در هندسه فینسلر-راندرز
    - تعبیه پس‌رو تاکنز با تطابق قطعی قیمت پایانی بازار
    - کالیبراسیون ناوردای راندرز با لدویت-ولف و نگاشت تحلیلی tanh
    - حل عددی RK4 همراه با گام تصویر پایستگی انرژی فینسلری
    - مقداردهی اولیه کاملاً قطعی از طریق بازگشت پوانکاره روی جاذب
    - بهینه‌سازی میانگین فرشه و کریدورهای نامتقارن با مقیاس صحیح قیمت
    """

    def __init__(self, ticker: str = "BZ=F", period: str = "7y", k_neighbors: int = 35):
        self.ticker = ticker
        self.period = period
        self.k_neighbors = k_neighbors

        self.df = pd.DataFrame()
        self.dates = pd.DatetimeIndex([])
        self.prices = np.array([])
        self.log_p = np.array([])
        self.s_dot = np.array([])

        self.tau = 4
        self.m = 3

        # ماتریس‌های کلاف مماس
        self.X = np.array([])  # x in M (موقعیت روی خمینه)
        self.Y = np.array([])  # y in TM (سرعت مماس)
        self.manifold_dates = pd.DatetimeIndex([])
        self.tree = None

        # نتایج ژئودزیک‌ها و میانگین فرشه
        self.geodesic_results = {}

    def fetch_and_reconstruct(self) -> "FinslerGeodesicSimulationEngine":
        """دریافت داده‌های رسمی بازار و ساخت کلاف مماس با تعبیه پس‌رو بدون تاخیر زمانی"""
        ticker_obj = yf.Ticker(self.ticker)
        raw = ticker_obj.history(period=self.period, interval="1d", auto_adjust=False)

        if raw.empty:
            raw = yf.download(self.ticker, period=self.period, interval="1d", auto_adjust=False)
            if isinstance(raw.columns, pd.MultiIndex):
                raw.columns = raw.columns.get_level_values(0)

        raw = raw[['Close', 'Volume']].dropna()
        raw = raw[(raw['Close'] > 0) & (raw['Volume'] > 0)].copy()

        self.df = raw
        self.dates = pd.to_datetime(raw.index)
        self.prices = raw['Close'].values
        self.log_p = np.log(self.prices)

        # مشتق تحلیلی سرعت مماس بدون شیفت فاز
        win = int(np.clip(len(self.log_p) // 100, 11, 25))
        if win % 2 == 0: win += 1
        self.s_dot = savgol_filter(self.log_p, window_length=win, polyorder=3, deriv=1)

        # استخراج تحلیلی تاخیر بهینه زمانی با AMI
        data = self.log_p
        iqr_val = max(iqr(data), 1e-6)
        n_bins = int(np.clip(np.ceil((data.max() - data.min()) / (2.0 * iqr_val * len(data)**(-1/3))), 16, 48))
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
        self.tau = int(local_min[0] + 1) if len(local_min) > 0 else 4

        # ساخت منیفلد M و کلاف مماس TM با تعبیه پس‌رو تاکنز
        N = len(self.log_p)
        tau = self.tau
        X_cols = [self.log_p[(2 - j) * tau: N - j * tau if j > 0 else N] for j in range(3)]
        Y_cols = [self.s_dot[(2 - j) * tau: N - j * tau if j > 0 else N] for j in range(3)]

        self.X = np.column_stack(X_cols)
        self.Y = np.column_stack(Y_cols)
        self.manifold_dates = self.dates[2 * tau:]
        self.tree = cKDTree(self.X)

        print(f"=== کلاف مماس آماده شد: {len(self.X)} گره | آخرین قیمت رسمی بازار: ${self.prices[-1]:.2f} ===")
        return self

    def _get_local_metrics(self, x_pt: np.ndarray):
        """استخراج تحلیلی کوواریانس لدویت-ولف و ۱-فرم باد نقدینگی با نگاشت tanh"""
        eps = 1e-12
        _, nearest_idx = self.tree.query(x_pt, k=1)
        anchor_x = self.X[nearest_idx]

        _, idxs = self.tree.query(anchor_x, k=self.k_neighbors)
        local_X = self.X[idxs]
        local_Y = self.Y[idxs]

        cov_lw, _ = ledoit_wolf(local_X)
        a_mat = np.linalg.pinv(cov_lw)
        a_inv = cov_lw

        w_cov = a_mat @ np.mean(local_Y, axis=0)
        w_norm = np.sqrt(np.maximum(w_cov @ a_inv @ w_cov, 0.0))
        b_vec = (np.tanh(w_norm) / (w_norm + eps)) * w_cov
        return a_mat, a_inv, b_vec, local_X

    def _compute_spray_coefficients(self, x_pt: np.ndarray, y_pt: np.ndarray):
        """محاسبه دقیق ضرایب اسپری باو-چرن-شن G^i با مهارگر منظم‌شده"""
        m = self.m
        eps = 1e-12
        a_mat, a_inv, b_vec, local_pts = self._get_local_metrics(x_pt)

        alpha_sq = y_pt @ a_mat @ y_pt
        if alpha_sq < 1e-14:
            return np.zeros(m), a_mat, b_vec, 1e-6

        alpha = np.sqrt(max(alpha_sq, eps))
        beta = float(np.dot(b_vec, y_pt))
        F_val = max(alpha + beta, 1e-6)

        # حل معادلات گرادیان با منظم‌سازی تیخونوف برای پایداری قطعی
        _, nearest_idx = self.tree.query(x_pt, k=1)
        anchor_x = self.X[nearest_idx]
        dX = local_pts - anchor_x
        reg = np.linalg.solve(dX.T @ dX + 1e-4 * np.eye(m), dX.T)

        dA = np.zeros((m, m, m))
        dB = np.zeros((m, m))
        for p_i, pt in enumerate(local_pts):
            a_p, _, b_p, _ = self._get_local_metrics(pt)
            diff_a = a_p - a_mat
            diff_b = b_p - b_vec
            for k in range(m):
                dA[:, :, k] += diff_a * reg[k, p_i]
                dB[:, k] += diff_b * reg[k, p_i]

        # نمادهای کریستوفل ریمانی: gamma^i_{jk}
        gamma = np.zeros((m, m, m))
        for i in range(m):
            for j in range(m):
                for k in range(m):
                    gamma[i, j, k] = 0.5 * np.sum(a_inv[i, :] * (dA[:, j, k] + dA[:, k, j] - dA[j, k, :]))

        gamma_00 = np.einsum('ijk,j,k->i', gamma, y_pt, y_pt)

        # مشتق هم‌وردا nabla_k b_i
        nabla_b = np.zeros((m, m))
        for i in range(m):
            for k in range(m):
                nabla_b[i, k] = dB[i, k] - np.dot(gamma[:, i, k], b_vec)

        r_mat = 0.5 * (nabla_b + nabla_b.T)
        s_mat = 0.5 * (nabla_b - nabla_b.T)

        r_00 = y_pt @ r_mat @ y_pt
        s_up = a_inv @ s_mat
        s_i_0 = s_up @ y_pt
        s_0 = np.dot(b_vec, s_i_0)

        P_val = (r_00 - 2.0 * alpha * s_0) / (4.0 * F_val)
        G_vec = 0.5 * gamma_00 + P_val * y_pt + 0.5 * alpha * s_i_0
        G_clipped = np.clip(G_vec, -30.0, 30.0)
        return G_clipped, a_mat, b_vec, F_val

    def _rk4_geodesic_step_with_projection(self, x: np.ndarray, y: np.ndarray, dt: float, target_F: float):
        """
        یک گام انتگرال‌گیری RK4 روی دستگاه هم‌زمان کلاف مماس
        همراه با گام تصویر تحلیلی پایستگی انرژی: F(x, y) = target_F
        """
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

        x_next = x + (dt / 6.0) * (k1_x + 2*k2_x + 2*k3_x + k4_x)
        y_next = y + (dt / 6.0) * (k1_y + 2*k2_y + 2*k3_y + k4_y)

        # گام تصویر هندسی: بازتنظیم اندازه بردار مماس به سطح انرژی ثابت اولیه
        _, a_mat, b_vec, _ = self._compute_spray_coefficients(x_next, y_next)

        alpha_next = np.sqrt(max(y_next @ a_mat @ y_next, 1e-12))
        beta_next = np.dot(b_vec, y_next)
        F_current = alpha_next + beta_next

        if F_current > 1e-6:
            y_next = y_next * (target_F / F_current)

        return x_next, y_next

    def simulate_poincare_geodesics(self, horizon: int = 25, k_scenarios: int = 7, dt: float = 0.20):
        """
        تولید سناریوهای قطعی بر مبنای همسایگان پوانکاره و محاسبه تحلیلی میانگین فرشه
        همراه با تبدیل مقیاس پایدار و تصحیح‌شده برای کریدورهای قیمت
        """
        N = len(self.X)
        x_now = self.X[-1].copy()

        # استخراج همسایگان پوانکاره با رعایت پنجره ناهمبستگی زمانی (حداقل ۳۰ روز قبل)
        decorrelation_window = 30
        _, candidate_indices = self.tree.query(x_now, k=min(k_scenarios * 4 + 10, N - 1))

        valid_indices = []
        for idx in candidate_indices:
            if (N - 1 - idx) > decorrelation_window:
                valid_indices.append(idx)
            if len(valid_indices) == k_scenarios:
                break

        if len(valid_indices) < k_scenarios:
            valid_indices = candidate_indices[1: k_scenarios + 1]

        trajectories = np.zeros((k_scenarios, horizon + 1, self.m))
        velocities = np.zeros((k_scenarios, horizon + 1, self.m))
        energies = np.zeros(k_scenarios)
        scenario_origins = []

        for k_idx, n_i in enumerate(valid_indices):
            x_sim = x_now.copy()
            y_sim = self.Y[n_i].copy()

            _, a_0, b_0, F_0 = self._compute_spray_coefficients(x_sim, y_sim)
            energies[k_idx] = F_0
            scenario_origins.append({
                'date': self.manifold_dates[n_i],
                'price': self.prices[n_i + 2 * self.tau],
                'energy': F_0
            })

            trajectories[k_idx, 0, :] = x_sim
            velocities[k_idx, 0, :] = y_sim

            for step in range(1, horizon + 1):
                for _ in range(5):  # ۵ زیرگام در روز برای دقت سمپلکتیک
                    x_sim, y_sim = self._rk4_geodesic_step_with_projection(x_sim, y_sim, dt, F_0)
                trajectories[k_idx, step, :] = x_sim
                velocities[k_idx, step, :] = y_sim

        # محاسبه تحلیلی میانگین فرشه در هر گام زمانی
        frechet_path = np.zeros((horizon + 1, self.m))
        frechet_std_up = np.zeros(horizon + 1)
        frechet_std_down = np.zeros(horizon + 1)

        for step in range(horizon + 1):
            pts_t = trajectories[:, step, :]

            # کمینه‌سازی مجموع مجذور فواصل راندرز
            def frechet_objective(p):
                a_mat, _, b_vec, _ = self._get_local_metrics(p)
                total_loss = 0.0
                for q in pts_t:
                    delta = q - p
                    alpha_d = np.sqrt(np.maximum(delta @ a_mat @ delta, 0.0))
                    beta_d = np.dot(b_vec, delta)
                    total_loss += (alpha_d + beta_d)**2
                return total_loss / k_scenarios

            init_guess = np.mean(pts_t, axis=0)
            opt_res = minimize(frechet_objective, init_guess, method='Nelder-Mead', options={'maxiter': 80})
            p_optimal = opt_res.x if opt_res.success else init_guess
            frechet_path[step] = p_optimal

            # استخراج ویژگی‌های متریک نقطه میانگین فرشه
            a_opt, _, b_opt, _ = self._get_local_metrics(p_optimal)
            sqrt_a11 = np.sqrt(max(a_opt[0, 0], 1e-6))
            b1 = b_opt[0]

            # محاسبه پراکندگی سناریوها مستقیماً در مختصات خطی لگاریتم قیمت (نه فاصله بیابانی ماهالانوبیس)
            delta_log_prices = pts_t[:, 0] - p_optimal[0]

            pos_deltas = delta_log_prices[delta_log_prices >= 0]
            neg_deltas = delta_log_prices[delta_log_prices < 0]
            total_dispersion = np.std(delta_log_prices) if len(delta_log_prices) > 1 else 0.015

            # ضرایب تصحیح اصطکاک باد راندرز در جهت صعود و نزول
            asymmetry_up = np.clip(1.0 / (1.0 + (b1 / sqrt_a11)), 0.6, 1.8)
            asymmetry_down = np.clip(1.0 / (1.0 - (b1 / sqrt_a11)), 0.6, 1.8)

            sigma_up = np.sqrt(np.mean(pos_deltas**2)) if len(pos_deltas) > 0 else total_dispersion
            sigma_down = np.sqrt(np.mean(neg_deltas**2)) if len(neg_deltas) > 0 else total_dispersion

            # مهار پراکندگی در محدوده‌های نوسان روزانه طبیعی (بین ۱٪ تا ۱۵٪ نوسان تجمعی)
            frechet_std_up[step] = np.clip(sigma_up * asymmetry_up, 0.012, 0.16)
            frechet_std_down[step] = np.clip(sigma_down * asymmetry_down, 0.012, 0.16)

        forecast_dates = pd.date_range(start=self.dates[-1], periods=horizon + 1, freq='B')

        # تبدیل صحیح به مقیاس دلار: exp(ln P +- delta_ln_P)
        frechet_price = np.exp(frechet_path[:, 0])
        upper_corridor = np.exp(frechet_path[:, 0] + frechet_std_up)
        lower_corridor = np.exp(frechet_path[:, 0] - frechet_std_down)

        self.geodesic_results = {
            'forecast_dates': forecast_dates,
            'trajectories': trajectories,
            'velocities': velocities,
            'energies': energies,
            'scenario_origins': scenario_origins,
            'frechet_path': frechet_path,
            'frechet_price': frechet_price,
            'upper_corridor': upper_corridor,
            'lower_corridor': lower_corridor,
            'initial_spot': self.prices[-1]
        }
        print(f"فرشه کالیبره شد | قیمت جاری: ${self.prices[-1]:.2f} | کریدور افق پایانی: [${lower_corridor[-1]:.2f} تا ${upper_corridor[-1]:.2f}]")
        return self

    # =========================================================
    # فضای حالت ۱: بادبزن ژئودزیک‌های قطعی و میانگین فرشه در منیفلد (M)
    # =========================================================
    def plot_geodesic_state_space_3d(self) -> go.Figure:
        """ترسیم سه‌بعدی خمینه جاذب، پرتاب ژئودزیک‌ها و میانگین فرشه با جزئیات کامل روی گره‌ها"""
        if not self.geodesic_results:
            raise ValueError("نخست متد simulate_poincare_geodesics را اجرا کنید.")

        res = self.geodesic_results
        trajs = res['trajectories']
        f_path = res['frechet_path']
        tau = self.tau

        fig = go.Figure()

        # ۱. خمینه تاریخی بازار (جاذب فاز)
        hist_len = min(600, len(self.X))
        sub_X = self.X[-hist_len:]
        fig.add_trace(go.Scatter3d(
            x=sub_X[:, 0], y=sub_X[:, 1], z=sub_X[:, 2],
            mode='lines',
            line=dict(color='rgba(150, 150, 150, 0.25)', width=1.5),
            hoverinfo='none',
            name='Historical Manifold Attractor'
        ))

        # ۲. رسم ژئودزیک‌های قطعی پوانکاره با گره‌های تفصیلی
        k_scenarios = trajs.shape[0]
        for k in range(k_scenarios):
            origin_info = res['scenario_origins'][k]
            f_energy = res['energies'][k]

            geo_hovers = [
                f"<b>Scenario:</b> #{k+1} (Poincaré Recurrence)<br>"
                f"<b>Historical Origin:</b> {origin_info['date'].strftime('%Y-%m-%d')} (${origin_info['price']:.2f})<br>"
                f"<b>Step:</b> T+{step} ({res['forecast_dates'][step].strftime('%Y-%m-%d')})<br>"
                f"<b>Price Level:</b> ${np.exp(trajs[k, step, 0]):.2f}<br>"
                f"<b>Coordinates:</b> [{trajs[k, step, 0]:.3f}, {trajs[k, step, 1]:.3f}, {trajs[k, step, 2]:.3f}]<br>"
                f"<b>Conserved Finsler Energy F:</b> {f_energy:.4f}"
                for step in range(trajs.shape[1])
            ]

            fig.add_trace(go.Scatter3d(
                x=trajs[k, :, 0], y=trajs[k, :, 1], z=trajs[k, :, 2],
                mode='lines+markers',
                line=dict(color='rgba(0, 240, 255, 0.45)', width=2),
                marker=dict(size=3, color='#00F0FF'),
                text=geo_hovers,
                hoverinfo='text',
                showlegend=(k == 0),
                name='Deterministic Poincare Geodesics'
            ))

        # ۳. مسیر میانگین فرشه تحلیلی (مرکز ثقل هندسی)
        frechet_hovers = [
            f"<b>Fréchet Center Node</b><br>"
            f"<b>Forecast Date:</b> {res['forecast_dates'][step].strftime('%Y-%m-%d')}<br>"
            f"<b>Canonical Price:</b> ${res['frechet_price'][step]:.2f}<br>"
            f"<b>Upper Corridor:</b> ${res['upper_corridor'][step]:.2f}<br>"
            f"<b>Lower Corridor:</b> ${res['lower_corridor'][step]:.2f}<br>"
            f"<b>Coordinates:</b> [{f_path[step, 0]:.3f}, {f_path[step, 1]:.3f}, {f_path[step, 2]:.3f}]"
            for step in range(len(f_path))
        ]

        fig.add_trace(go.Scatter3d(
            x=f_path[:, 0], y=f_path[:, 1], z=f_path[:, 2],
            mode='lines+markers',
            line=dict(color='#39FF14', width=5),
            marker=dict(size=5, color='#FFFFFF', symbol='circle'),
            text=frechet_hovers,
            hoverinfo='text',
            name='Exact Fréchet Mean Trajectory'
        ))

        # ۴. گره نقطه آغازین بازار
        fig.add_trace(go.Scatter3d(
            x=[self.X[-1, 0]], y=[self.X[-1, 1]], z=[self.X[-1, 2]],
            mode='markers+text',
            marker=dict(size=10, color='#FFE600', symbol='diamond'),
            text=[f"Current Spot: ${res['initial_spot']:.2f}"],
            textposition="top center",
            name='Current Market State (Launch Node)'
        ))

        fig.update_layout(
            title=f"<b>Module 4 State Space: Deterministic Geodesic Flow & Fréchet Center ({self.ticker})</b><br>Symplectic RK4 with Finsler Energy Projection Step F(x, y) = const",
            template='plotly_dark',
            height=850,
            scene=dict(
                xaxis=dict(title='x¹: s(t) = ln Close(t)', backgroundcolor="#111111"),
                yaxis=dict(title=f'x²: s(t - {tau})', backgroundcolor="#111111"),
                zaxis=dict(title=f'x³: s(t - {2*tau})', backgroundcolor="#111111")
            ),
            margin=dict(l=0, r=0, b=0, t=60)
        )
        return fig

    # =========================================================
    # فضای حالت ۲: تصویر زمانی قیمت و کریدورهای نامتقارن فینسلری
    # =========================================================
    def plot_price_corridors(self, hist_bars: int = 90) -> go.Figure:
        """ترسیم کریدورهای عدم‌قطعیت نامتقارن قیمت و میانگین فرشه در مقیاس واقعی دلار"""
        if not self.geodesic_results:
            raise ValueError("نخست متد simulate_poincare_geodesics را اجرا کنید.")

        res = self.geodesic_results
        trajs = res['trajectories']
        k_scenarios = trajs.shape[0]

        fig = go.Figure()

        # داده‌های تاریخی واقعی بازار
        fig.add_trace(go.Scatter(
            x=self.dates[-hist_bars:], y=self.prices[-hist_bars:],
            mode='lines', line=dict(color='#A0AAB2', width=1.5),
            name='Observed Market Spot'
        ))

        # خطوط بادبزن ژئودزیک‌های قطعی پوانکاره
        for k in range(k_scenarios):
            fig.add_trace(go.Scatter(
                x=res['forecast_dates'], y=np.exp(trajs[k, :, 0]),
                mode='lines', line=dict(color='rgba(0, 240, 255, 0.25)', width=1),
                showlegend=(k == 0), name='Deterministic Geodesics'
            ))

        # کریدور نامتقارن بالادست و پایین‌دست
        fig.add_trace(go.Scatter(
            x=res['forecast_dates'], y=res['upper_corridor'],
            mode='lines', line=dict(color='rgba(255, 230, 0, 0.7)', width=1.2, dash='dash'),
            name='Finsler Asymmetric Upper Corridor'
        ))
        fig.add_trace(go.Scatter(
            x=res['forecast_dates'], y=res['lower_corridor'],
            mode='lines', line=dict(color='rgba(255, 49, 49, 0.7)', width=1.2, dash='dash'),
            fill='tonexty', fillcolor='rgba(255, 230, 0, 0.08)',
            name='Finsler Asymmetric Lower Corridor'
        ))

        # مسیر میانگین فرشه
        fig.add_trace(go.Scatter(
            x=res['forecast_dates'], y=res['frechet_price'],
            mode='lines+markers', line=dict(color='#39FF14', width=2.5),
            marker=dict(size=4), name='Fréchet Mean Geodesic Trajectory'
        ))

        fig.update_layout(
            title=f"<b>Module 4: Geodesic Price Fan & Asymmetric Finsler Corridors for {self.ticker}</b><br>Initial Spot: ${res['initial_spot']:.2f} | Scaled Deterministic Forecast Ensembles",
            xaxis=dict(title='Timeline'),
            yaxis=dict(title='Brent Crude Price ($/bbl)'),
            template='plotly_dark',
            height=650,
            showlegend=True
        )
        return fig


if __name__ == "__main__":
    # ۱. راه‌اندازی و بازسازی کلاف مماس
    engine_m4 = FinslerGeodesicSimulationEngine(ticker="BZ=F", period="7y", k_neighbors=35)
    engine_m4.fetch_and_reconstruct()

    # ۲. اجرای شبیه‌سازی ژئودزیک‌ها با مقیاس صحیح و بدون پرتاب عددی
    engine_m4.simulate_poincare_geodesics(horizon=7, k_scenarios=9, dt=0.20)

    # ۳. رسم فضای حالت سه‌بعدی منیفلد (ژئودزیک‌ها و میانگین فرشه)
    fig_geo_3d = engine_m4.plot_geodesic_state_space_3d()
    fig_geo_3d.show()

    # ۴. رسم کریدورهای قیمت بر حسب دلار واقعی نفت برنت
    fig_corridors = engine_m4.plot_price_corridors(hist_bars=90)
    fig_corridors.show()

    # ذخیره خروجی‌ها در قالب HTML تعاملی
    fig_geo_3d.write_html("module4_state_space_geodesics.html", include_plotlyjs='cdn')
    fig_corridors.write_html("module4_corrected_price_corridors.html", include_plotlyjs='cdn')
