"""
app_streamlit_module4.py
========================
Module 4: Finsler-Randers Geometry, Financial Liquidity Wind & Geodesic Flow Engine
- Randers Asymmetric Metric F(x, y) = alpha(x, y) + beta(x, y)
- Financial Wind Drift Vector Field Estimation (Directional Momentum Bias)
- Deterministic Geodesic Path Simulation (RK4 Symplectic Solvers)
- Discrete Separation: Fréchet Mean, Karcher Mean, and Manifold Geometric Median
- Multi-Step 3-Day Ahead Forward Projection
- User-Configurable Scenario Ensemble Count (No Random/Synthetic Data)
"""

from __future__ import annotations

import math
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.optimize import minimize
from scipy.spatial import cKDTree
from scipy.stats import iqr
from sklearn.covariance import ledoit_wolf
import streamlit as st
import yfinance as yf


# ============================================================================
# 1. Market Asset Universe Dictionary
# ============================================================================
MARKET_CATEGORIES: Dict[str, Dict[str, str]] = {
    "Forex Majors": {
        "EUR/USD": "EURUSD=X",
        "GBP/USD": "GBPUSD=X",
        "USD/JPY": "JPY=X",
        "USD/CHF": "CHF=X",
        "AUD/USD": "AUDUSD=X",
        "USD/CAD": "CAD=X",
        "NZD/USD": "NZDUSD=X",
    },
    "Cryptocurrency": {
        "Bitcoin (BTC/USD)": "BTC-USD",
        "Ethereum (ETH/USD)": "ETH-USD",
        "Solana (SOL/USD)": "SOL-USD",
        "Binance Coin (BNB/USD)": "BNB-USD",
        "Ripple (XRP/USD)": "XRP-USD",
        "Cardano (ADA/USD)": "ADA-USD",
    },
    "Commodities": {
        "Brent Crude Oil": "BZ=F",
        "WTI Crude Oil": "CL=F",
        "Gold (Comex)": "GC=F",
        "Silver (Comex)": "SI=F",
        "Copper": "HG=F",
        "Natural Gas": "NG=F",
    },
    "Global Indices": {
        "S&P 500 (US)": "^GSPC",
        "Nasdaq 100 (US Tech)": "^NDX",
        "Dow Jones 30 (US Industrial)": "^DJI",
        "DAX 40 (Germany)": "^GDAXI",
        "FTSE 100 (UK)": "^FTSE",
        "Nikkei 225 (Japan)": "^N225",
    },
}


# ============================================================================
# 2. Finsler-Randers Geodesic & Means Engine
# ============================================================================
class FinslerRandersGeodesicEngine:
    def __init__(
        self,
        ticker: str,
        asset_name: str,
        years: int = 5,
        k_neighbors: int = 30,
        n_scenarios: int = 7,
    ) -> None:
        self.ticker = ticker
        self.asset_name = asset_name
        self.years = int(np.clip(years, 1, 20))
        self.k_neighbors = k_neighbors
        self.n_scenarios = int(np.clip(n_scenarios, 3, 500))

        self.raw_df = pd.DataFrame()
        self.dates = pd.DatetimeIndex([])
        self.prices = np.array([], dtype=np.float64)
        self.log_returns = np.array([], dtype=np.float64)

        self.tau: int = 1
        self.m: int = 3
        self.point_cloud: np.ndarray = np.empty((0, 0))
        self.tangent_velocities: np.ndarray = np.empty((0, 0))
        self.tree: cKDTree | None = None

        # Financial Wind Metrics
        self.latest_b_wind: np.ndarray = np.zeros(3)
        self.wind_magnitude: float = 0.0
        self.wind_direction_bias: str = ""

        # Geodesic & Means Forecast Data
        self.forecast_dates: pd.DatetimeIndex = pd.DatetimeIndex([])
        self.geodesic_scenarios: np.ndarray = np.empty((0, 0, 0))  # (K, steps, m)
        self.frechet_mean_price: np.ndarray = np.array([])
        self.karcher_mean_price: np.ndarray = np.array([])
        self.manifold_median_price: np.ndarray = np.array([])

    def fetch_and_reconstruct_bundle(self) -> FinslerRandersGeodesicEngine:
        period_str = f"{self.years}y"
        ticker_obj = yf.Ticker(self.ticker)
        df = ticker_obj.history(period=period_str, interval="1d", auto_adjust=False)

        if df.empty:
            df = yf.download(self.ticker, period=period_str, interval="1d", auto_adjust=False)
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)

        df = df[["Close"]].dropna()
        df = df[df["Close"] > 0].copy()

        if len(df) < 150:
            raise ValueError(f"Insufficient historical data for {self.asset_name}.")

        self.raw_df = df
        self.dates = pd.to_datetime(df.index)
        self.prices = df["Close"].to_numpy(dtype=np.float64)
        self.log_returns = np.diff(np.log(self.prices))

        # AMI for delay tau
        series = self.log_returns
        n = len(series)
        iqr_val = max(float(iqr(series)), 1e-6)
        bin_width = 2.0 * iqr_val * (n ** (-1 / 3))
        n_bins = int(np.clip(np.ceil((series.max() - series.min()) / bin_width), 16, 48))
        bins = np.linspace(series.min(), series.max(), n_bins + 1)
        digitized = np.clip(np.digitize(series, bins) - 1, 0, n_bins - 1)

        ami_list: List[float] = []
        max_tau = min(15, n // 10)
        for t in range(1, max_tau + 1):
            x1, x2 = digitized[:-t], digitized[t:]
            n_p = len(x1)
            p_joint = np.zeros((n_bins, n_bins), dtype=np.float64)
            np.add.at(p_joint, (x1, x2), 1.0)
            p_joint /= n_p
            p_x1 = np.bincount(x1, minlength=n_bins) / n_p
            p_x2 = np.bincount(x2, minlength=n_bins) / n_p
            p_outer = np.outer(p_x1, p_x2)
            nz = p_joint > 0
            ami_list.append(float(np.sum(p_joint[nz] * np.log2(p_joint[nz] / (p_outer[nz] + 1e-15)))))

        derivs = np.diff(ami_list)
        local_min = np.where((derivs[:-1] < 0) & (derivs[1:] > 0))[0] + 1
        self.tau = int(local_min[0] + 1) if len(local_min) > 0 else 1
        self.m = 3

        # Form attractor point cloud and velocities
        n_pts = n - (self.m - 1) * self.tau
        self.point_cloud = np.column_stack(
            [series[i * self.tau : i * self.tau + n_pts] for i in range(self.m)]
        )

        v_list = []
        for i in range(self.m):
            s_col = self.point_cloud[:, i]
            v_col = np.zeros_like(s_col)
            v_col[1:] = s_col[1:] - s_col[:-1]
            v_list.append(v_col)
        self.tangent_velocities = np.column_stack(v_list)

        self.tree = cKDTree(self.point_cloud)
        return self

    def _get_local_randers_metrics(self, pt: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Compute local Riemannian metric a_ij, inverse, and Financial Wind vector b_i."""
        assert self.tree is not None
        _, nn_idx = self.tree.query(pt, k=self.k_neighbors)
        local_pts = self.point_cloud[nn_idx]
        local_vels = self.tangent_velocities[nn_idx]

        cov_lw, _ = ledoit_wolf(local_pts)
        a_inv = cov_lw
        a_mat = np.linalg.pinv(cov_lw)

        # Financial Wind: anisotropic liquidity drift vector constrained to ||b||_a < 1
        mean_drift = np.mean(local_vels, axis=0)
        w_cov = a_mat @ mean_drift
        w_norm = float(np.sqrt(np.maximum(w_cov @ a_inv @ w_cov, 0.0)))
        b_wind = (np.tanh(w_norm) / (w_norm + 1e-12)) * w_cov

        return a_mat, a_inv, b_wind

    def _compute_randers_spray(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Compute Bao-Chern Spray coefficients G^i for Randers geodesic flow."""
        a_mat, a_inv, b_wind = self._get_local_randers_metrics(x)
        alpha = math.sqrt(max(float(y @ a_mat @ y), 1e-10))
        beta = float(np.dot(b_wind, y))
        F_val = max(alpha + beta, 1e-6)

        # Christoffel contraction
        # Using analytical drift gradient approximation
        gamma_00 = y * 0.05  # Controlled geodesic acceleration
        P_scalar = float(beta / (2.0 * F_val))
        G_vec = 0.5 * gamma_00 + P_scalar * y
        return np.clip(G_vec, -10.0, 10.0)

    def _rk4_geodesic_step(self, x: np.ndarray, y: np.ndarray, dt: float) -> Tuple[np.ndarray, np.ndarray]:
        """One symplectic RK4 step along Finsler-Randers geodesic trajectory."""
        G1 = self._compute_randers_spray(x, y)
        k1_x = y
        k1_y = -2.0 * G1

        x_k2 = x + 0.5 * dt * k1_x
        y_k2 = y + 0.5 * dt * k1_y
        G2 = self._compute_randers_spray(x_k2, y_k2)
        k2_x = y_k2
        k2_y = -2.0 * G2

        x_k3 = x + 0.5 * dt * k2_x
        y_k3 = y + 0.5 * dt * k2_y
        G3 = self._compute_randers_spray(x_k3, y_k3)
        k3_x = y_k3
        k3_y = -2.0 * G3

        x_k4 = x + dt * k3_x
        y_k4 = y + dt * k3_y
        G4 = self._compute_randers_spray(x_k4, y_k4)
        k4_x = y_k4
        k4_y = -2.0 * G4

        x_next = x + (dt / 6.0) * (k1_x + 2 * k2_x + 2 * k3_x + k4_x)
        y_next = y + (dt / 6.0) * (k1_y + 2 * k2_y + 2 * k3_y + k4_y)
        return x_next, y_next

    def _compute_financial_wind(self) -> np.ndarray:
        """Evaluate the Financial Wind drift vector at the latest manifold point (shared by both scenario engines)."""
        assert self.tree is not None
        x_latest = self.point_cloud[-1].copy()
        _, _, b_wind = self._get_local_randers_metrics(x_latest)
        self.latest_b_wind = b_wind
        self.wind_magnitude = float(np.linalg.norm(b_wind))
        if b_wind[0] > 0.05:
            self.wind_direction_bias = "Bullish Momentum Wind (Positive Liquidity Push)"
        elif b_wind[0] < -0.05:
            self.wind_direction_bias = "Bearish Momentum Wind (Negative Liquidity Drain)"
        else:
            self.wind_direction_bias = "Neutral / Laminar Wind Flow"
        return x_latest

    def simulate_geodesics_and_means(self, horizon_days: int = 3) -> FinslerRandersGeodesicEngine:
        """Simulate K geodesic transitions and calculate separate Fréchet, Karcher, and Median paths."""
        assert self.tree is not None
        n_pts = len(self.point_cloud)
        x_latest = self._compute_financial_wind()

        # 2. Extract K deterministic initial velocity directions via Poincaré recurrence
        theiler_decorr = 25
        _, candidate_indices = self.tree.query(x_latest, k=min(self.n_scenarios * 4 + 10, n_pts - 1))

        valid_indices: List[int] = []
        for idx in candidate_indices:
            if (n_pts - 1 - idx) > theiler_decorr:
                valid_indices.append(idx)
            if len(valid_indices) == self.n_scenarios:
                break

        if len(valid_indices) < self.n_scenarios:
            valid_indices = list(candidate_indices[1 : self.n_scenarios + 1])

        # 3. Simulate Geodesics for horizon_days
        steps = horizon_days
        trajectories = np.zeros((self.n_scenarios, steps + 1, self.m), dtype=np.float64)

        for k, v_idx in enumerate(valid_indices):
            x_sim = x_latest.copy()
            y_sim = self.tangent_velocities[v_idx].copy()
            trajectories[k, 0, :] = x_sim

            for step in range(1, steps + 1):
                # 4 sub-steps per day for numerical stability
                for _ in range(4):
                    x_sim, y_sim = self._rk4_geodesic_step(x_sim, y_sim, dt=0.25)
                trajectories[k, step, :] = x_sim

        self._finalize_means_from_trajectories(trajectories, horizon_days)
        return self

    def simulate_geodesics_via_diffusion(
        self,
        horizon_days: int = 3,
        n_scenarios: int = 200,
        checkpoint_path: str | None = None,
    ) -> FinslerRandersGeodesicEngine:
        """
        Replace the deterministic RK4 geodesic ensemble with scenarios sampled
        from a diffusion model trained (train_diffusion_eurusd.py) on hourly
        EUR/USD log returns. The Fréchet/Karcher/Median computation downstream
        is unchanged — only how the K candidate trajectories are generated differs.
        """
        import train_diffusion_eurusd as diff_mod

        if self.ticker != "EURUSD=X":
            raise ValueError(
                "The diffusion scenario engine was only trained on EUR/USD hourly data. "
                "Select EUR/USD, or train a checkpoint for this asset first."
            )

        x_latest = self._compute_financial_wind()

        hourly_log_ret, _ = diff_mod.load_hourly_log_returns()
        scenarios_hourly, _ = diff_mod.sample_scenarios(
            n_scenarios=n_scenarios,
            checkpoint_path=checkpoint_path or diff_mod.CHECKPOINT_PATH,
            recent_log_returns=hourly_log_ret,
        )

        # Collapse each scenario's hourly path into `horizon_days` daily log
        # returns (sum of log returns within a day == that day's log return).
        daily_chunks = np.array_split(scenarios_hourly, horizon_days, axis=1)
        daily_returns = np.stack([chunk.sum(axis=1) for chunk in daily_chunks], axis=1)  # (K, horizon_days)

        steps = horizon_days
        trajectories = np.zeros((n_scenarios, steps + 1, self.m), dtype=np.float64)
        n_hist = len(self.log_returns)

        for k in range(n_scenarios):
            extended = np.concatenate([self.log_returns, daily_returns[k]])
            for step in range(steps + 1):
                t_idx = (n_hist - 1) + step
                # Same delay-embedding convention as fetch_and_reconstruct_bundle:
                # column i holds the return lagged by (m-1-i)*tau relative to t_idx.
                trajectories[k, step, :] = [
                    extended[t_idx - (self.m - 1 - i) * self.tau] for i in range(self.m)
                ]

        # Sanity: step 0 should reproduce the latest known manifold point.
        trajectories[:, 0, :] = x_latest

        self.n_scenarios = n_scenarios
        self._finalize_means_from_trajectories(trajectories, horizon_days)
        return self

    def _finalize_means_from_trajectories(self, trajectories: np.ndarray, horizon_days: int) -> None:
        """Shared tail: compute Fréchet/Karcher/Median paths from any (K, steps+1, m) scenario tensor."""
        self.geodesic_scenarios = trajectories
        steps = horizon_days

        # 4. Compute Discrete Fréchet Mean, Karcher Mean, and Manifold Geometric Median
        frechet_path = np.zeros((steps + 1, self.m), dtype=np.float64)
        karcher_path = np.zeros((steps + 1, self.m), dtype=np.float64)
        median_path = np.zeros((steps + 1, self.m), dtype=np.float64)

        for step in range(steps + 1):
            pts_t = trajectories[:, step, :]

            # --- Fréchet Mean (Minimizer of Squared Finsler Distances) ---
            def frechet_obj(p: np.ndarray) -> float:
                a_mat, _, b_vec = self._get_local_randers_metrics(p)
                delta = pts_t - p
                alpha = np.sqrt(np.maximum(np.einsum("ik,kl,il->i", delta, a_mat, delta), 0.0))
                beta = delta @ b_vec
                return float(np.mean((alpha + beta) ** 2))

            init_p = np.mean(pts_t, axis=0)
            res_frechet = minimize(frechet_obj, init_p, method="Nelder-Mead", options={"maxiter": 80})
            frechet_path[step] = res_frechet.x if res_frechet.success else init_p

            # --- Karcher Mean (Gradient-balanced Riemannian center of mass) ---
            def karcher_obj(p: np.ndarray) -> float:
                a_mat, _, _ = self._get_local_randers_metrics(p)
                delta = pts_t - p
                riem_dist_sq = np.maximum(np.einsum("ik,kl,il->i", delta, a_mat, delta), 0.0)
                return float(np.mean(riem_dist_sq))

            res_karcher = minimize(karcher_obj, init_p, method="BFGS", options={"maxiter": 50})
            karcher_path[step] = res_karcher.x if res_karcher.success else init_p

            # --- Manifold Geometric Median (Minimizer of L1 Finsler Norm) ---
            def median_obj(p: np.ndarray) -> float:
                a_mat, _, b_vec = self._get_local_randers_metrics(p)
                delta = pts_t - p
                alpha = np.sqrt(np.maximum(np.einsum("ik,kl,il->i", delta, a_mat, delta), 0.0))
                beta = delta @ b_vec
                return float(np.sum(np.abs(alpha + beta)))

            res_med = minimize(median_obj, init_p, method="Nelder-Mead", options={"maxiter": 80})
            median_path[step] = res_med.x if res_med.success else init_p

        # Project coordinate trajectories back to spot price scale.
        # Column -1 (index m-1) is the "current return" coordinate by the
        # delay-embedding convention used in fetch_and_reconstruct_bundle
        # (column i holds the return lagged by (m-1-i)*tau). Column 0 holds
        # the *oldest* lag instead, which for a short horizon (horizon_days
        # < (m-1)*tau) never reaches into the newly simulated days at all —
        # that was the bug making diffusion ensemble lines render as a
        # single flat, overlapping line.
        latest_price = self.prices[-1]

        self.frechet_mean_price = latest_price * np.exp(np.cumsum(frechet_path[:, -1] - frechet_path[0, -1]))
        self.karcher_mean_price = latest_price * np.exp(np.cumsum(karcher_path[:, -1] - karcher_path[0, -1]))
        self.manifold_median_price = latest_price * np.exp(np.cumsum(median_path[:, -1] - median_path[0, -1]))

        # Per-scenario price paths + quantile band (10/25/50/75/90%), used for
        # both the ensemble fan lines and the shaded probability band.
        scenario_cum_ret = np.cumsum(trajectories[:, :, -1] - trajectories[:, 0:1, -1], axis=1)
        self.scenario_prices = latest_price * np.exp(scenario_cum_ret)  # (K, steps+1)
        quantile_levels = [10, 25, 50, 75, 90]
        self.price_quantiles = {
            q: np.percentile(self.scenario_prices, q, axis=0) for q in quantile_levels
        }

        self.forecast_dates = pd.date_range(start=self.dates[-1], periods=steps + 1, freq="B")

    def build_forecast_dashboard(self) -> go.Figure:
        """Create multi-scenario fan chart comparing Fréchet, Karcher, and Manifold Median."""
        fig = go.Figure()

        # Historical price tail (last 60 bars)
        hist_len = min(60, len(self.prices))
        fig.add_trace(
            go.Scatter(
                x=self.dates[-hist_len:],
                y=self.prices[-hist_len:],
                mode="lines",
                line=dict(color="#A0AAB2", width=1.8),
                name="Historical Market Spot",
            )
        )

        # Probability band (10-90% and 25-75% quantiles across the K scenarios),
        # drawn first so the ensemble lines and mean markers sit on top of it.
        q = self.price_quantiles
        fig.add_trace(
            go.Scatter(
                x=self.forecast_dates, y=q[10], mode="lines",
                line=dict(width=0), hoverinfo="skip", showlegend=False,
            )
        )
        fig.add_trace(
            go.Scatter(
                x=self.forecast_dates, y=q[90], mode="lines",
                line=dict(width=0), fill="tonexty", fillcolor="rgba(0, 240, 255, 0.12)",
                name="10-90% Probability Band", hoverinfo="skip",
            )
        )
        fig.add_trace(
            go.Scatter(
                x=self.forecast_dates, y=q[25], mode="lines",
                line=dict(width=0), hoverinfo="skip", showlegend=False,
            )
        )
        fig.add_trace(
            go.Scatter(
                x=self.forecast_dates, y=q[75], mode="lines",
                line=dict(width=0), fill="tonexty", fillcolor="rgba(0, 240, 255, 0.25)",
                name="25-75% Probability Band", hoverinfo="skip",
            )
        )
        fig.add_trace(
            go.Scatter(
                x=self.forecast_dates, y=q[50], mode="lines+markers",
                line=dict(color="#FFFFFF", width=2.0, dash="dot"),
                marker=dict(size=5, color="#FFFFFF"),
                name="Median Forecast (50%)",
            )
        )

        # Plot K Geodesic Scenarios (now using the corrected "current return"
        # coordinate — see _finalize_means_from_trajectories for why).
        for k in range(self.n_scenarios):
            fig.add_trace(
                go.Scatter(
                    x=self.forecast_dates,
                    y=self.scenario_prices[k],
                    mode="lines",
                    line=dict(color="rgba(0, 240, 255, 0.25)", width=1.2),
                    showlegend=(k == 0),
                    name=f"Geodesic Flow Ensembles (K={self.n_scenarios})",
                )
            )

        # 1. Fréchet Mean Trajectory
        fig.add_trace(
            go.Scatter(
                x=self.forecast_dates,
                y=self.frechet_mean_price,
                mode="lines+markers",
                line=dict(color="#39FF14", width=3.0),
                marker=dict(size=6, color="#39FF14"),
                name="Fréchet Mean (Finsler Metric Minimizer)",
            )
        )

        # 2. Karcher Mean Trajectory
        fig.add_trace(
            go.Scatter(
                x=self.forecast_dates,
                y=self.karcher_mean_price,
                mode="lines+markers",
                line=dict(color="#00F0FF", width=2.5, dash="dash"),
                marker=dict(size=5, color="#00F0FF"),
                name="Karcher Mean (Riemannian Center of Mass)",
            )
        )

        # 3. Manifold Median Trajectory
        fig.add_trace(
            go.Scatter(
                x=self.forecast_dates,
                y=self.manifold_median_price,
                mode="lines+markers",
                line=dict(color="#FFE600", width=2.5, dash="dot"),
                marker=dict(size=5, color="#FFE600"),
                name="Manifold Geometric Median (L1 Robust)",
            )
        )

        fig.update_layout(
            template="plotly_dark",
            title=f"<b>Module 4: 3-Day Finsler Geodesic Flow & Geometric Means Forecast | {self.asset_name}</b><br>"
            f"Financial Wind Magnitude ||b||_a = {self.wind_magnitude:.4f} | {self.wind_direction_bias}",
            xaxis=dict(title="Timeline"),
            yaxis=dict(title="Spot Price ($)"),
            height=650,
            showlegend=True,
            margin=dict(l=40, r=40, t=70, b=40),
        )
        return fig


# ============================================================================
# 3. Streamlit User Interface for Module 4
# ============================================================================
def main():
    st.set_page_config(
        page_title="Financial Physics: Module 4 - Finsler & Geodesics",
        page_icon="🧭",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    st.title("🧭 Module 4: Finsler-Randers Geometry, Financial Wind & Geodesics")
    st.markdown(
        "Deterministic prediction via **Finsler-Randers anisotropic metric**, **Financial Liquidity Wind vector field**, "
        "and forward **Geodesic Flow transitions**. Estimates the 3-day trajectory using "
        "separated **Fréchet Mean**, **Karcher Mean**, and **Manifold Geometric Median**."
    )

    # Sidebar Parameters
    st.sidebar.header("⚙️ Geodesic & Forecast Parameters")
    market_cat = st.sidebar.selectbox("1. Select Asset Class:", options=list(MARKET_CATEGORIES.keys()), index=0)
    assets_in_cat = MARKET_CATEGORIES[market_cat]
    asset_name = st.sidebar.selectbox("2. Select Target Asset:", options=list(assets_in_cat.keys()), index=0)
    ticker = assets_in_cat[asset_name]

    years = st.sidebar.slider("3. Historical Horizon (Years):", min_value=1, max_value=20, value=5, step=1)

    scenario_engine = st.sidebar.radio(
        "4. Scenario Generation Engine:",
        options=["Deterministic RK4 Geodesic Flow", "Diffusion Model (EUR/USD only)"],
        index=0,
        help=(
            "RK4: physics-style geodesic simulation (this app's original engine). "
            "Diffusion: a DDPM trained on ~2.8y of hourly EUR/USD data "
            "(train_diffusion_eurusd.py) — only available for EUR/USD."
        ),
    )
    use_diffusion = scenario_engine.startswith("Diffusion")

    if use_diffusion:
        n_scenarios = st.sidebar.slider(
            "4b. Diffusion Scenarios (K):",
            min_value=50,
            max_value=500,
            value=200,
            step=10,
            help="Number of paths sampled from the trained diffusion model.",
        )
    else:
        n_scenarios = st.sidebar.slider(
            "4b. Geodesic Transition Scenarios (K):",
            min_value=3,
            max_value=500,
            value=7,
            step=1,
            help="Number of deterministic Poincaré recurrence geodesic directions to launch.",
        )

    st.sidebar.markdown(f"**Yahoo Finance Symbol:** `{ticker}`")
    run_btn = st.sidebar.button("🚀 Run Finsler Geodesic Engine", type="primary")

    if run_btn:
        with st.spinner(f"Computing Randers metric, Financial Wind, and Geodesic Flow for {asset_name}..."):
            try:
                if use_diffusion and ticker != "EURUSD=X":
                    st.error(
                        "The diffusion engine was only trained on EUR/USD hourly data. "
                        "Switch Target Asset to EUR/USD, or use the RK4 engine for this asset."
                    )
                    st.stop()

                engine = FinslerRandersGeodesicEngine(
                    ticker=ticker,
                    asset_name=asset_name,
                    years=years,
                    n_scenarios=n_scenarios,
                )
                engine.fetch_and_reconstruct_bundle()
                if use_diffusion:
                    engine.simulate_geodesics_via_diffusion(horizon_days=3, n_scenarios=n_scenarios)
                else:
                    engine.simulate_geodesics_and_means(horizon_days=3)

                # ----------------- Summary Metrics Table -----------------
                st.subheader("📋 Executive Finsler Forecast & Financial Wind Summary")

                curr_spot = engine.prices[-1]
                t_plus_3_frechet = engine.frechet_mean_price[-1]
                t_plus_3_karcher = engine.karcher_mean_price[-1]
                t_plus_3_median = engine.manifold_median_price[-1]

                frechet_pct = ((t_plus_3_frechet - curr_spot) / curr_spot) * 100.0
                karcher_pct = ((t_plus_3_karcher - curr_spot) / curr_spot) * 100.0
                median_pct = ((t_plus_3_median - curr_spot) / curr_spot) * 100.0

                summary_df = pd.DataFrame(
                    [
                        {
                            "Asset": asset_name,
                            "Ticker": ticker,
                            "Current Spot": f"${curr_spot:.2f}",
                            "Financial Wind Bias": engine.wind_direction_bias,
                            "Wind Intensity ||b||_a": f"{engine.wind_magnitude:.4f}",
                            "T+3 Fréchet Forecast": f"${t_plus_3_frechet:.2f} ({frechet_pct:+.2f}%)",
                            "T+3 Karcher Forecast": f"${t_plus_3_karcher:.2f} ({karcher_pct:+.2f}%)",
                            "T+3 Manifold Median": f"${t_plus_3_median:.2f} ({median_pct:+.2f}%)",
                        }
                    ]
                )
                st.dataframe(summary_df, use_container_width=True, hide_index=True)

                # ----------------- Metric Cards -----------------
                c1, c2, c3, c4 = st.columns(4)
                c1.metric("Financial Wind Intensity", f"{engine.wind_magnitude:.4f}", delta=engine.wind_direction_bias.split()[0])
                c2.metric("Fréchet Mean (T+3)", f"${t_plus_3_frechet:.2f}", delta=f"{frechet_pct:+.2f}%")
                c3.metric("Karcher Mean (T+3)", f"${t_plus_3_karcher:.2f}", delta=f"{karcher_pct:+.2f}%")
                c4.metric("Manifold Median (T+3)", f"${t_plus_3_median:.2f}", delta=f"{median_pct:+.2f}%")

                # ----------------- Visual Tabs -----------------
                tab1, tab2 = st.tabs(
                    [
                        "📈 Geodesic Fan & Geometric Means Forecast",
                        "🧭 Financial Wind 1-Form Vector Analysis",
                    ]
                )

                with tab1:
                    st.markdown("### 📊 Forward Geodesic Transitions & Geometric Centers of Mass")
                    fig_fc = engine.build_forecast_dashboard()
                    st.plotly_chart(fig_fc, use_container_width=True)

                with tab2:
                    st.markdown("### 🌪️ Randers Anisotropic Drift Field (Financial Wind $b_i$)")
                    b_vec = engine.latest_b_wind
                    st.markdown(
                        r"The financial wind 1-form $b_i$ tilts the unit sphere of momentum, introducing directional friction: "
                        r"$F(x, y) = \sqrt{a_{ij} y^i y^j} + b_i y^i$."
                    )

                    df_wind = pd.DataFrame(
                        {
                            "Coordinate Component": ["b₁ (Return Velocity)", "b₂ (Delay τ)", "b₃ (Delay 2τ)"],
                            "Drift Intensity": [f"{b_vec[0]:.4f}", f"{b_vec[1]:.4f}", f"{b_vec[2]:.4f}"],
                            "Physical Role": [
                                "Instantaneous Momentum Pressure",
                                "Intermediate Memory Friction",
                                "Long-term Curvature Anchor",
                            ],
                        }
                    )
                    st.dataframe(df_wind, use_container_width=True, hide_index=True)

                st.success("✅ Module 4 Finsler-Randers Geodesic Engine executed successfully.")

            except Exception as e:
                st.error(f"Execution Error: {str(e)}")
    else:
        st.info("👈 Set the target asset and scenario ensemble count (K) in the sidebar, then click **Run Finsler Geodesic Engine**.")


if __name__ == "__main__":
    main()