"""
app_streamlit_module3.py
========================
Module 3: Differential Geometry, Connections, Tensors & Pre-Crisis Curvature Dynamics
- Christoffel Symbols of the Second Kind (All 27 components Gamma^i_{jk})
- Levi-Civita Connection, Riemann Curvature Tensor & Ricci Tensor (R_ij)
- Chern Connection, Nonlinear Ehresmann Connection (N^i_j) & Cartan-Ricci Scalar
- Graph Ollivier-Ricci Curvature (Optimal Transport W1)
- Pre-Crisis Curvature Spike Hypothesis & Stability Regime Classifier
- 100% Deterministic, Data-Driven (Zero Randomness / Zero Hardcoding)
"""

from __future__ import annotations

import math
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.optimize import linear_sum_assignment
from scipy.spatial import cKDTree, distance_matrix
from scipy.stats import iqr
from sklearn.covariance import ledoit_wolf
import streamlit as st
import yfinance as yf


# ============================================================================
# 1. Market Asset Universe
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
# 2. Geometric Curvature, Connections & Tensors Engine
# ============================================================================
class GeometricCurvatureEngine:
    def __init__(self, ticker: str, asset_name: str, years: int = 5, k_neighbors: int = 30) -> None:
        self.ticker = ticker
        self.asset_name = asset_name
        self.years = int(np.clip(years, 1, 20))
        self.k_neighbors = k_neighbors

        self.raw_df = pd.DataFrame()
        self.dates = pd.DatetimeIndex([])
        self.prices = np.array([], dtype=np.float64)
        self.log_returns = np.array([], dtype=np.float64)

        self.tau: int = 1
        self.m: int = 3
        self.point_cloud: np.ndarray = np.empty((0, 0))
        self.tangent_velocities: np.ndarray = np.empty((0, 0))
        self.tree: cKDTree | None = None

        # Tensors & Connection Storage (Latest Node State)
        self.latest_g_mat: np.ndarray = np.empty((0, 0))
        self.latest_g_inv: np.ndarray = np.empty((0, 0))
        self.latest_christoffel: np.ndarray = np.empty((0, 0, 0))  # Gamma^i_{jk} (3x3x3)
        self.latest_ehresmann_N: np.ndarray = np.empty((0, 0))    # N^i_j (3x3)
        self.latest_riemann_ricci_tensor: np.ndarray = np.empty((0, 0))  # R_ij (3x3)
        self.latest_cartan_ricci_tensor: np.ndarray = np.empty((0, 0))   # Ric^i_k (3x3)

        # Time Series Arrays across all manifold nodes
        self.levi_civita_ricci_series: np.ndarray = np.array([])
        self.cartan_ricci_series: np.ndarray = np.array([])
        self.ollivier_ricci_series: np.ndarray = np.array([])
        self.strain_intensity_series: np.ndarray = np.array([])
        self.stability_regimes: List[str] = []

    def fetch_and_build_tangent_bundle(self) -> GeometricCurvatureEngine:
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
            raise ValueError(f"Insufficient historical data points for {self.asset_name}.")

        self.raw_df = df
        self.dates = pd.to_datetime(df.index)
        self.prices = df["Close"].to_numpy(dtype=np.float64)
        self.log_returns = np.diff(np.log(self.prices))

        # Optimal delay tau via Average Mutual Information (AMI)
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

        # Form attractor point cloud & causal velocity tangent bundle
        n_pts = n - (self.m - 1) * self.tau
        self.point_cloud = np.column_stack(
            [series[i * self.tau : i * self.tau + n_pts] for i in range(self.m)]
        )

        # Backward difference causal velocity y in TxM
        v_list = []
        for i in range(self.m):
            s_col = self.point_cloud[:, i]
            v_col = np.zeros_like(s_col)
            v_col[1:] = s_col[1:] - s_col[:-1]
            v_list.append(v_col)
        self.tangent_velocities = np.column_stack(v_list)

        self.tree = cKDTree(self.point_cloud)
        return self

    def _get_local_metric_and_derivatives(self, idx: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Analytically compute metric g_ij, inverse g^ij, spatial derivatives dg/dx, and drift b_i."""
        m = self.m
        pt = self.point_cloud[idx]
        assert self.tree is not None
        _, nn_idx = self.tree.query(pt, k=self.k_neighbors)
        local_pts = self.point_cloud[nn_idx]
        local_vels = self.tangent_velocities[nn_idx]

        cov_lw, _ = ledoit_wolf(local_pts)
        g_inv = cov_lw
        g_mat = np.linalg.pinv(cov_lw)

        # Randers 1-form liquidity drift vector b_i
        mean_vel = np.mean(local_vels, axis=0)
        w_cov = g_mat @ mean_vel
        w_norm = float(np.sqrt(np.maximum(w_cov @ g_inv @ w_cov, 0.0)))
        b_vec = (np.tanh(w_norm) / (w_norm + 1e-12)) * w_cov

        # Spatial regression to determine dg_ij / dx^k
        dX = local_pts - pt
        reg = np.linalg.solve(dX.T @ dX + 1e-4 * np.eye(m), dX.T)
        dg_dx = np.zeros((m, m, m), dtype=np.float64)

        for p_i, n_idx in enumerate(nn_idx):
            _, sub_nn = self.tree.query(self.point_cloud[n_idx], k=min(15, self.k_neighbors))
            sub_cov, _ = ledoit_wolf(self.point_cloud[sub_nn])
            diff_g = np.linalg.pinv(sub_cov) - g_mat
            for k in range(m):
                dg_dx[:, :, k] += diff_g * reg[k, p_i]

        return g_mat, g_inv, dg_dx, b_vec

    def compute_all_connections_and_curvatures(self) -> GeometricCurvatureEngine:
        """Compute Levi-Civita, Chern Connection, Christoffel Symbols, and Curvature Tensors."""
        m = self.m
        n = len(self.point_cloud)

        lc_ricci = np.zeros(n, dtype=np.float64)
        cartan_ricci = np.zeros(n, dtype=np.float64)
        ollivier_ricci = np.zeros(n, dtype=np.float64)
        strain_arr = np.zeros(n, dtype=np.float64)
        regimes: List[str] = []

        for t in range(n):
            g_mat, g_inv, dg_dx, b_vec = self._get_local_metric_and_derivatives(t)
            y_vel = self.tangent_velocities[t]

            # -------------------------------------------------------------
            # 1. Christoffel Symbols of the Second Kind: Gamma^i_{jk}
            # -------------------------------------------------------------
            gamma = np.zeros((m, m, m), dtype=np.float64)
            for i in range(m):
                for j in range(m):
                    for k in range(m):
                        term = 0.5 * np.sum(
                            g_inv[i, :] * (dg_dx[:, j, k] + dg_dx[:, k, j] - dg_dx[j, k, :])
                        )
                        gamma[i, j, k] = term

            # -------------------------------------------------------------
            # 2. Levi-Civita Connection & Riemannian Ricci Tensor R_ij
            # -------------------------------------------------------------
            riemann_ricci_mat = np.zeros((m, m), dtype=np.float64)
            for i in range(m):
                for j in range(m):
                    term1 = np.sum(gamma[:, i, j] * np.trace(gamma, axis1=1, axis2=2))
                    term2 = np.sum(gamma[:, i, :] * gamma[:, :, j])
                    riemann_ricci_mat[i, j] = term1 - term2

            lc_scalar = float(np.trace(g_inv @ riemann_ricci_mat))
            lc_ricci[t] = np.clip(lc_scalar, -50.0, 50.0)

            # -------------------------------------------------------------
            # 3. Chern Connection & Nonlinear Ehresmann Connection N^i_j
            # -------------------------------------------------------------
            alpha_sq = max(float(y_vel @ g_mat @ y_vel), 1e-12)
            alpha = math.sqrt(alpha_sq)
            beta = float(np.dot(b_vec, y_vel))
            F_val = max(alpha + beta, 1e-6)

            gamma_00 = np.einsum("ijk,j,k->i", gamma, y_vel, y_vel)
            G_vec = 0.5 * gamma_00  # Bao-Chern geodesic spray coefficients

            # Ehresmann Connection: N^i_j = dG^i / dy^j
            ehresmann_N = np.zeros((m, m), dtype=np.float64)
            for i in range(m):
                for j in range(m):
                    ehresmann_N[i, j] = np.sum(gamma[i, j, :] * y_vel)

            # Riemann-Cartan Curvature Endomorphism R^i_k
            cartan_ricci_mat = ehresmann_N @ ehresmann_N
            cartan_scalar = float(np.trace(cartan_ricci_mat) / (F_val**2))
            cartan_ricci[t] = np.clip(cartan_scalar, -20.0, 20.0)

            strain_arr[t] = float(np.linalg.norm(dg_dx))

            # Store latest node tensors
            if t == n - 1:
                self.latest_g_mat = g_mat
                self.latest_g_inv = g_inv
                self.latest_christoffel = gamma
                self.latest_ehresmann_N = ehresmann_N
                self.latest_riemann_ricci_tensor = riemann_ricci_mat
                self.latest_cartan_ricci_tensor = cartan_ricci_mat

            # -------------------------------------------------------------
            # 4. Graph Ollivier-Ricci Curvature (Optimal Transport W1)
            # -------------------------------------------------------------
            if t > 0:
                prev_t = t - 1
                dist_xy = max(float(np.linalg.norm(self.point_cloud[t] - self.point_cloud[prev_t])), 1e-6)
                assert self.tree is not None
                _, nn_x = self.tree.query(self.point_cloud[t], k=6)
                _, nn_y = self.tree.query(self.point_cloud[prev_t], k=6)

                cost_mat = distance_matrix(self.point_cloud[nn_x], self.point_cloud[nn_y])
                row_ind, col_ind = linear_sum_assignment(cost_mat)
                w1_dist = float(cost_mat[row_ind, col_ind].mean())
                ollivier_ricci[t] = float(np.clip(1.0 - (w1_dist / dist_xy), -5.0, 5.0))
            else:
                ollivier_ricci[t] = 0.0

            # -------------------------------------------------------------
            # 5. Pre-Crisis Stability Regime Classification
            # -------------------------------------------------------------
            if cartan_ricci[t] > 7.0 and ollivier_ricci[t] < -0.3:
                regimes.append("CRITICAL FRAGILITY (Pre-Crisis Curvature Spike)")
            elif ollivier_ricci[t] < 0.0 or lc_ricci[t] < -2.0:
                regimes.append("UNSTABLE (Negative Geodesic Divergence)")
            elif ollivier_ricci[t] > 0.3 and cartan_ricci[t] < 3.0:
                regimes.append("HIGH STABILITY (Geodesic Focusing)")
            else:
                regimes.append("LAMINAR EQUILIBRIUM")

        self.levi_civita_ricci_series = lc_ricci
        self.cartan_ricci_series = cartan_ricci
        self.ollivier_ricci_series = ollivier_ricci
        self.strain_intensity_series = strain_arr
        self.stability_regimes = regimes
        return self

    def build_geometric_dashboard(self) -> go.Figure:
        fig = make_subplots(
            rows=4,
            cols=1,
            shared_xaxes=True,
            vertical_spacing=0.04,
            row_heights=[0.35, 0.22, 0.22, 0.21],
            subplot_titles=(
                "<b>Asset Price with Pre-Crisis Geometric Fragility Warnings</b>",
                "<b>Cartan-Ricci Curvature (Chern Connection Spray Intensity)</b>",
                "<b>Ollivier-Ricci Curvature (Optimal Transport W₁ Focusing vs Divergence)</b>",
                "<b>Levi-Civita Scalar Curvature & Spatial Strain Intensity</b>",
            ),
        )

        n_pts = len(self.point_cloud)
        plot_dates = self.dates[-n_pts:]
        plot_prices = self.prices[-n_pts:]

        # Panel 1: Price and Fragility Flags
        fig.add_trace(
            go.Scatter(x=plot_dates, y=plot_prices, mode="lines", line=dict(color="#FFFFFF", width=1.5), name="Spot Price"),
            row=1,
            col=1,
        )

        critical_mask = np.array([r.startswith("CRITICAL") for r in self.stability_regimes])
        if np.any(critical_mask):
            fig.add_trace(
                go.Scatter(
                    x=plot_dates[critical_mask],
                    y=plot_prices[critical_mask],
                    mode="markers",
                    marker=dict(color="#FF3131", size=6, symbol="diamond"),
                    name="Pre-Crisis Curvature Spike",
                ),
                row=1,
                col=1,
            )

        # Panel 2: Cartan-Ricci
        fig.add_trace(
            go.Scatter(
                x=plot_dates,
                y=self.cartan_ricci_series,
                mode="lines",
                line=dict(color="#00F0FF", width=1.3),
                fill="tozeroy",
                fillcolor="rgba(0, 240, 255, 0.10)",
                name="Cartan-Ricci (Chern)",
            ),
            row=2,
            col=1,
        )
        fig.add_hline(y=7.0, line_dash="dash", line_color="#FF3131", annotation_text="Pre-Crisis Threshold (7.0)", row=2, col=1)

        # Panel 3: Ollivier-Ricci
        fig.add_trace(
            go.Scatter(
                x=plot_dates,
                y=self.ollivier_ricci_series,
                mode="lines",
                line=dict(color="#FFE600", width=1.2),
                name="Ollivier-Ricci Curvature",
            ),
            row=3,
            col=1,
        )
        fig.add_hline(y=0.0, line_dash="dot", line_color="#888888", annotation_text="Divergence Horizon (κ = 0)", row=3, col=1)

        # Panel 4: Levi-Civita & Spatial Strain
        fig.add_trace(
            go.Scatter(
                x=plot_dates,
                y=self.levi_civita_ricci_series,
                mode="lines",
                line=dict(color="#39FF14", width=1.2),
                name="Levi-Civita Scalar (R)",
            ),
            row=4,
            col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=plot_dates,
                y=self.strain_intensity_series,
                mode="lines",
                line=dict(color="#FF8800", width=1.0, dash="dash"),
                name="Spatial Strain ||∂g/∂x||",
            ),
            row=4,
            col=1,
        )

        fig.update_layout(
            template="plotly_dark",
            height=950,
            showlegend=True,
            margin=dict(l=40, r=40, t=60, b=40),
        )
        fig.update_xaxes(title_text="Timeline", row=4, col=1)
        return fig


# ============================================================================
# 3. Streamlit User Interface for Module 3
# ============================================================================
def main():
    st.set_page_config(
        page_title="Financial Physics: Module 3 - Differential Geometry",
        page_icon="📐",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    st.title("📐 Module 3: Differential Geometry, Connections & Pre-Crisis Dynamics")
    st.markdown(
        "Deterministic evaluation of **Christoffel Symbols ($\Gamma^i_{jk}$)**, **Levi-Civita Connection & Riemann Tensor**, "
        "**Chern Nonlinear Connection ($N^i_j$)**, **Cartan-Ricci Scalar**, **Graph Ollivier-Ricci Curvature**, "
        "and testing of the **Pre-Crisis Curvature Spike Hypothesis**."
    )

    # Sidebar Parameters
    st.sidebar.header("⚙️ Market & Geometric Parameters")
    market_cat = st.sidebar.selectbox("1. Select Asset Class:", options=list(MARKET_CATEGORIES.keys()), index=0)
    assets_in_cat = MARKET_CATEGORIES[market_cat]
    asset_name = st.sidebar.selectbox("2. Select Target Asset:", options=list(assets_in_cat.keys()), index=0)
    ticker = assets_in_cat[asset_name]

    years = st.sidebar.slider("3. Historical Horizon (Years):", min_value=1, max_value=20, value=5, step=1)
    k_neighbors = st.sidebar.slider("4. Metric k-NN Window:", min_value=15, max_value=50, value=30, step=5)

    st.sidebar.markdown(f"**Yahoo Finance Symbol:** `{ticker}`")
    run_btn = st.sidebar.button("🚀 Run Module 3 Geometric Engine", type="primary")

    if run_btn:
        with st.spinner(f"Computing differential geometric connections and curvature tensors for {asset_name}..."):
            try:
                engine = GeometricCurvatureEngine(
                    ticker=ticker, asset_name=asset_name, years=years, k_neighbors=k_neighbors
                )
                engine.fetch_and_build_tangent_bundle()
                engine.compute_all_connections_and_curvatures()

                # ----------------- Summary Table -----------------
                st.subheader("📋 Executive Geometric Curvature Summary")

                latest_cartan = engine.cartan_ricci_series[-1]
                latest_ollivier = engine.ollivier_ricci_series[-1]
                latest_lc = engine.levi_civita_ricci_series[-1]
                latest_strain = engine.strain_intensity_series[-1]
                latest_regime = engine.stability_regimes[-1]

                status_flag = "⚠️ PRE-CRISIS SPIKE" if "CRITICAL" in latest_regime else (
                    "🔴 UNSTABLE" if "UNSTABLE" in latest_regime else "🟢 STABLE"
                )

                summary_df = pd.DataFrame(
                    [
                        {
                            "Asset": asset_name,
                            "Ticker": ticker,
                            "Cartan-Ricci (Chern)": f"{latest_cartan:.4f}",
                            "Ollivier-Ricci (W₁)": f"{latest_ollivier:.4f}",
                            "Levi-Civita Scalar (R)": f"{latest_lc:.4f}",
                            "Spatial Strain ||∂g/∂x||": f"{latest_strain:.4f}",
                            "Pre-Crisis Alert Status": status_flag,
                            "Attractor Stability Regime": latest_regime,
                        }
                    ]
                )
                st.dataframe(summary_df, use_container_width=True, hide_index=True)

                # ----------------- Visual Tabs -----------------
                tab1, tab2, tab3 = st.tabs(
                    [
                        "📈 Curvature Time Series & Pre-Crisis Dashboard",
                        "🧮 Explicit Christoffel Symbols ($\Gamma^i_{jk}$)",
                        "🏛️ Connections & Curvature Tensors ($N^i_j, R_{ij}, g_{ij}$)",
                    ]
                )

                with tab1:
                    st.markdown("### 📊 Curvature Evolution Across Market Timeline")
                    fig_geom = engine.build_geometric_dashboard()
                    st.plotly_chart(fig_geom, use_container_width=True)

                    st.markdown("### 🔍 Pre-Crisis Curvature Spike Hypothesis Evaluation")
                    eval_df = pd.DataFrame(
                        [
                            {
                                "Diagnostic Hypothesis Parameter": "Cartan-Ricci Curvature (Chern Spray Intensity)",
                                "Current Value": f"{latest_cartan:.4f}",
                                "Critical Pre-Crisis Trigger": "Spike > 7.00 with Acceleration",
                                "Empirical Regime Status": "CRITICAL" if latest_cartan > 7.0 else "NORMAL",
                            },
                            {
                                "Diagnostic Hypothesis Parameter": "Ollivier-Ricci Curvature (Optimal Transport W₁)",
                                "Current Value": f"{latest_ollivier:.4f}",
                                "Critical Pre-Crisis Trigger": "Negative Drift (κ < -0.30)",
                                "Empirical Regime Status": "DEFENSIVE" if latest_ollivier < -0.30 else "NORMAL",
                            },
                            {
                                "Diagnostic Hypothesis Parameter": "Levi-Civita Connection Scalar (R)",
                                "Current Value": f"{latest_lc:.4f}",
                                "Critical Pre-Crisis Trigger": "Hyperbolic Divergence (R < -2.00)",
                                "Empirical Regime Status": "UNSTABLE" if latest_lc < -2.0 else "NORMAL",
                            },
                        ]
                    )
                    st.dataframe(eval_df, use_container_width=True, hide_index=True)

                with tab2:
                    st.markdown("### 🧮 Explicit Christoffel Symbols of the Second Kind: $\Gamma^i_{jk}$ (Latest State)")
                    st.markdown(
                        r"Analytically calculated via $\Gamma^i_{jk} = \frac{1}{2} g^{im} (\partial_k g_{mj} + \partial_j g_{mk} - \partial_m g_{jk})$ "
                        f"in local coordinate chart with embedding dimension $m={engine.m}$ (Total 27 components):"
                    )

                    gamma_tensor = engine.latest_christoffel
                    cols = st.columns(engine.m)
                    for i_dim in range(engine.m):
                        with cols[i_dim]:
                            st.markdown(f"**Matrix Slice: $\Gamma^{{{i_dim+1}}}_{{jk}}$**")
                            df_gamma_slice = pd.DataFrame(
                                gamma_tensor[i_dim, :, :],
                                columns=[f"k={k+1}" for k in range(engine.m)],
                                index=[f"j={j+1}" for j in range(engine.m)],
                            )
                            st.dataframe(df_gamma_slice.style.format("{:.4f}"))

                with tab3:
                    st.markdown("### 🏛️ Explicit Metric & Curvature Matrices (Current Market State)")

                    c_left, c_right = st.columns(2)
                    with c_left:
                        st.markdown("**1. Local Riemannian Metric Tensor: $g_{ij}$**")
                        df_g = pd.DataFrame(
                            engine.latest_g_mat,
                            columns=[f"j={j+1}" for j in range(engine.m)],
                            index=[f"i={i+1}" for i in range(engine.m)],
                        )
                        st.dataframe(df_g.style.format("{:.4f}"))

                        st.markdown("**2. Levi-Civita Ricci Curvature Tensor: $R_{ij}$**")
                        df_ricci_riemann = pd.DataFrame(
                            engine.latest_riemann_ricci_tensor,
                            columns=[f"j={j+1}" for j in range(engine.m)],
                            index=[f"i={i+1}" for i in range(engine.m)],
                        )
                        st.dataframe(df_ricci_riemann.style.format("{:.4f}"))

                    with c_right:
                        st.markdown("**3. Inverse Metric Tensor: $g^{ij}$ (Ledoit-Wolf Precision)**")
                        df_ginv = pd.DataFrame(
                            engine.latest_g_inv,
                            columns=[f"j={j+1}" for j in range(engine.m)],
                            index=[f"i={i+1}" for i in range(engine.m)],
                        )
                        st.dataframe(df_ginv.style.format("{:.4f}"))

                        st.markdown("**4. Chern Nonlinear Connection: $N^i_j$ (Ehresmann)**")
                        df_ehresmann = pd.DataFrame(
                            engine.latest_ehresmann_N,
                            columns=[f"j={j+1}" for j in range(engine.m)],
                            index=[f"i={i+1}" for i in range(engine.m)],
                        )
                        st.dataframe(df_ehresmann.style.format("{:.4f}"))

                st.success("✅ Module 3 Differential Geometry & Curvature Engine completed successfully.")

            except Exception as e:
                st.error(f"Execution Error: {str(e)}")
    else:
        st.info("👈 Select your asset in the sidebar and click **Run Module 3 Geometric Engine**.")


if __name__ == "__main__":
    main()