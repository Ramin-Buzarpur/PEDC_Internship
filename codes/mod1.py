"""
app_streamlit.py
================
Nonlinear Dynamics & Chaos Analysis Dashboard (Module 1)
- Deterministic Phase Space Reconstruction (AMI & FNN Curves)
- 2D / 3D Phase Delay Reconstruction Plots
- Largest Lyapunov Exponent (Rosenstein LLE) with IAAFT Surrogate Testing
- Pesin's Identity Kolmogorov-Sinai Metric Entropy
- Clean English UI & Interactive Streamlit Dashboard
"""

from __future__ import annotations

import math
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.spatial import cKDTree
from scipy.stats import iqr
import streamlit as st
import yfinance as yf


# ============================================================================
# 1. Asset Universe Dictionary
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
# 2. IAAFT Surrogate Generation Algorithm
# ============================================================================
def generate_iaaft_surrogates(
    series: np.ndarray, n_surrogates: int = 20, max_iter: int = 100
) -> List[np.ndarray]:
    """Iterated Amplitude Adjusted Fourier Transform (IAAFT) Surrogate Test."""
    n = len(series)
    sorted_orig = np.sort(series)
    fft_orig_amp = np.abs(np.fft.rfft(series))

    surrogates = []
    for k in range(n_surrogates):
        rng = np.random.RandomState(42 + k)
        random_phases = rng.uniform(0, 2 * np.pi, len(fft_orig_amp))
        random_phases[0] = 0.0
        if n % 2 == 0:
            random_phases[-1] = 0.0

        s_fft = fft_orig_amp * np.exp(1j * random_phases)
        s_curr = np.fft.irfft(s_fft, n=n)

        for _ in range(max_iter):
            ranks = np.argsort(np.argsort(s_curr))
            s_ampl = sorted_orig[ranks]
            s_ampl_fft = np.fft.rfft(s_ampl)
            phases = np.angle(s_ampl_fft)
            s_new_fft = fft_orig_amp * np.exp(1j * phases)
            s_curr = np.fft.irfft(s_new_fft, n=n)

        ranks = np.argsort(np.argsort(s_curr))
        surrogates.append(sorted_orig[ranks])

    return surrogates


# ============================================================================
# 3. Nonlinear Dynamics & Chaos Engine
# ============================================================================
class AdvancedChaosEntropyEngine:
    def __init__(self, ticker: str, asset_name: str, years: int = 10) -> None:
        self.ticker = ticker
        self.asset_name = asset_name
        self.years = int(np.clip(years, 1, 20))
        self.raw_df = pd.DataFrame()
        self.dates = pd.DatetimeIndex([])
        self.prices = np.array([], dtype=np.float64)
        self.log_returns = np.array([], dtype=np.float64)

        # Embedding & Delay diagnostics
        self.tau: int = 1
        self.m: int = 3
        self.ami_tau_grid: np.ndarray = np.array([])
        self.ami_scores: np.ndarray = np.array([])
        self.fnn_m_grid: np.ndarray = np.array([])
        self.fnn_percentages: np.ndarray = np.array([])
        self.embedded_matrix: np.ndarray = np.empty((0, 0))

        # Output metrics
        self.metrics: Dict[str, float] = {}
        self.lyapunov_divergence_curve: Tuple[np.ndarray, np.ndarray] = (np.array([]), np.array([]))
        self.iaaft_lle_distribution: List[float] = []
        self.iaaft_p_value: float = 1.0
        self.is_chaos_verified: bool = False

    def fetch_data(self) -> AdvancedChaosEntropyEngine:
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
        return self

    def reconstruct_phase_space(self) -> AdvancedChaosEntropyEngine:
        series = self.log_returns
        n = len(series)

        # 1. Average Mutual Information (AMI)
        iqr_val = max(float(iqr(series)), 1e-6)
        bin_width = 2.0 * iqr_val * (n ** (-1 / 3))
        n_bins = int(np.clip(np.ceil((series.max() - series.min()) / bin_width), 16, 48))
        bins = np.linspace(series.min(), series.max(), n_bins + 1)
        digitized = np.clip(np.digitize(series, bins) - 1, 0, n_bins - 1)

        ami_list: List[float] = []
        max_tau = min(15, n // 10)
        self.ami_tau_grid = np.arange(1, max_tau + 1)

        for t in self.ami_tau_grid:
            x1, x2 = digitized[:-t], digitized[t:]
            n_p = len(x1)
            p_joint = np.zeros((n_bins, n_bins), dtype=np.float64)
            np.add.at(p_joint, (x1, x2), 1.0)
            p_joint /= n_p

            p_x1 = np.bincount(x1, minlength=n_bins) / n_p
            p_x2 = np.bincount(x2, minlength=n_bins) / n_p
            p_outer = np.outer(p_x1, p_x2)

            nz = p_joint > 0
            ami = np.sum(p_joint[nz] * np.log2(p_joint[nz] / (p_outer[nz] + 1e-15)))
            ami_list.append(float(ami))

        self.ami_scores = np.array(ami_list)
        derivatives = np.diff(self.ami_scores)
        local_min = np.where((derivatives[:-1] < 0) & (derivatives[1:] > 0))[0] + 1
        self.tau = int(local_min[0] + 1) if len(local_min) > 0 else 1

        # 2. False Nearest Neighbors (FNN)
        tau = self.tau
        std_val = float(np.std(series))
        self.fnn_m_grid = np.arange(1, 7)
        fnn_list: List[float] = []
        m_optimal = 3

        for m in self.fnn_m_grid:
            n_vectors = n - m * tau
            if n_vectors <= 0:
                fnn_list.append(0.0)
                continue
            y_m = np.column_stack([series[i * tau : i * tau + n_vectors] for i in range(m)])
            tree = cKDTree(y_m)
            dists, idxs = tree.query(y_m, k=2)
            r_m = dists[:, 1]
            nn_idx = idxs[:, 1]

            delta_next = np.abs(series[m * tau : m * tau + n_vectors] - series[nn_idx + m * tau])
            r_next = np.sqrt(r_m**2 + delta_next**2)

            fnn_pct = float(
                np.mean(
                    (delta_next / (r_m + 1e-12) > 15.0) | (r_next / (std_val + 1e-12) > 2.0)
                )
                * 100.0
            )
            fnn_list.append(fnn_pct)

        self.fnn_percentages = np.array(fnn_list)
        valid_m = self.fnn_m_grid[self.fnn_percentages <= 5.0]
        self.m = int(valid_m[0]) if len(valid_m) > 0 else 3

        # Tangent Phase Space
        n_pts = n - (self.m - 1) * self.tau
        self.embedded_matrix = np.column_stack(
            [series[i * self.tau : i * self.tau + n_pts] for i in range(self.m)]
        )
        return self

    @staticmethod
    def _compute_lle_from_series(
        series: np.ndarray, m: int, tau: int, max_t: int = 25
    ) -> Tuple[float, np.ndarray, np.ndarray]:
        n = len(series)
        n_pts = n - (m - 1) * tau
        if n_pts < max_t + 15:
            return 0.0, np.array([]), np.array([])

        y = np.column_stack([series[i * tau : i * tau + n_pts] for i in range(m)])
        tree = cKDTree(y)
        theiler_window = max(tau * 2, 5)
        dists, idxs = tree.query(y, k=min(25, len(y) - 1))

        nn_indices = np.zeros(len(y), dtype=int)
        valid_mask = np.zeros(len(y), dtype=bool)

        for i in range(len(y)):
            for d, idx in zip(dists[i, 1:], idxs[i, 1:]):
                if abs(idx - i) > theiler_window and d > 1e-12:
                    nn_indices[i] = idx
                    valid_mask[i] = True
                    break

        valid_idx = np.where(valid_mask)[0]
        valid_idx = valid_idx[valid_idx + max_t < len(y)]
        valid_idx = valid_idx[nn_indices[valid_idx] + max_t < len(y)]

        if len(valid_idx) == 0:
            return 0.0, np.array([]), np.array([])

        div_curve = np.zeros(max_t, dtype=np.float64)
        for t in range(max_t):
            diffs = y[valid_idx + t] - y[nn_indices[valid_idx] + t]
            dists_t = np.linalg.norm(diffs, axis=1)
            dists_t = dists_t[dists_t > 0]
            div_curve[t] = np.mean(np.log(dists_t)) if len(dists_t) > 0 else 0.0

        t_steps = np.arange(max_t)
        fit_len = min(10, max_t // 2)
        slope = (
            float(np.polyfit(t_steps[1:fit_len], div_curve[1:fit_len], 1)[0])
            if fit_len > 2
            else 0.0
        )
        return slope, t_steps, div_curve

    def compute_lyapunov_and_iaaft_test(self, n_surrogates: int = 20) -> AdvancedChaosEntropyEngine:
        # 1. Original LLE
        lle_orig, t_steps, div_curve = self._compute_lle_from_series(
            self.log_returns, self.m, self.tau
        )
        self.metrics["Max_Lyapunov_Exponent"] = float(lle_orig)
        self.lyapunov_divergence_curve = (t_steps, div_curve)

        # 2. IAAFT Surrogates
        surrogates = generate_iaaft_surrogates(self.log_returns, n_surrogates=n_surrogates)
        surr_lles = []
        for s in surrogates:
            s_lle, _, _ = self._compute_lle_from_series(s, self.m, self.tau)
            surr_lles.append(s_lle)

        self.iaaft_lle_distribution = surr_lles

        # 3. Statistical p-value
        count_exceed = sum(1 for s_val in surr_lles if s_val >= lle_orig)
        self.iaaft_p_value = float((count_exceed + 1.0) / (n_surrogates + 1.0))
        self.is_chaos_verified = bool((lle_orig > 0.005) and (self.iaaft_p_value < 0.05))
        return self

    def compute_entropies(self) -> AdvancedChaosEntropyEngine:
        data = self.log_returns
        n = len(data)

        # 1. Shannon Entropy
        iqr_val = max(float(iqr(data)), 1e-6)
        n_bins = int(np.clip(np.ceil((data.max() - data.min()) / (2.0 * iqr_val * (n ** (-1 / 3)))), 16, 64))
        counts, _ = np.histogram(data, bins=n_bins)
        p = counts / np.sum(counts)
        p = p[p > 0]
        self.metrics["Shannon_Entropy"] = float(-np.sum(p * np.log2(p)))

        # 2. Sample Entropy (SampEn)
        m_s = 2
        r_thresh = 0.20 * float(np.std(data))

        def _count_pairs(m_len: int) -> float:
            vecs = np.column_stack([data[i : n - m_len + 1 + i] for i in range(m_len)])
            n_v = len(vecs)
            tree_s = cKDTree(vecs)
            pairs = tree_s.count_neighbors(tree_s, r=r_thresh, p=np.inf)
            return (pairs - n_v) / (n_v * (n_v - 1))

        c_m = _count_pairs(m_s)
        c_m1 = _count_pairs(m_s + 1)
        self.metrics["Sample_Entropy"] = float(-np.log(c_m1 / c_m)) if (c_m > 0 and c_m1 > 0) else 0.0

        # 3. Permutation Entropy (PE)
        d_p = 4
        patterns: Dict[Tuple[int, ...], int] = {}
        for i in range(n - d_p + 1):
            perm = tuple(np.argsort(data[i : i + d_p]))
            patterns[perm] = patterns.get(perm, 0) + 1

        probs_perm = np.array(list(patterns.values()), dtype=np.float64) / (n - d_p + 1)
        pe_raw = -np.sum(probs_perm * np.log2(probs_perm))
        self.metrics["Permutation_Entropy"] = float(pe_raw / np.log2(math.factorial(d_p)))

        # 4. Kolmogorov-Sinai Metric Entropy via Pesin's Theorem
        lle = self.metrics.get("Max_Lyapunov_Exponent", 0.0)
        pesin_ks = max(0.0, lle) if self.is_chaos_verified else 0.0
        self.metrics["Kolmogorov_Sinai_Pesin"] = float(pesin_ks)
        return self

    # =========================================================================
    # Diagnostic Plots
    # =========================================================================
    def build_embedding_diagnostics_figure(self) -> go.Figure:
        """Plot AMI, FNN, and 2D Phase Delay Plane."""
        fig = make_subplots(
            rows=1,
            cols=3,
            subplot_titles=(
                f"<b>Average Mutual Information (AMI) | Optimal τ* = {self.tau}</b>",
                f"<b>False Nearest Neighbors (FNN) | Optimal m* = {self.m}</b>",
                f"<b>Phase Delay Map: r(t) vs r(t - {self.tau})</b>",
            ),
            horizontal_spacing=0.08,
        )

        # AMI Plot
        fig.add_trace(
            go.Scatter(
                x=self.ami_tau_grid,
                y=self.ami_scores,
                mode="lines+markers",
                line=dict(color="#00F0FF", width=2),
                marker=dict(size=6),
                name="AMI(τ)",
            ),
            row=1,
            col=1,
        )
        fig.add_vline(
            x=self.tau,
            line_dash="dash",
            line_color="#FFE600",
            annotation_text=f"τ* = {self.tau}",
            row=1,
            col=1,
        )

        # FNN Plot
        fig.add_trace(
            go.Scatter(
                x=self.fnn_m_grid,
                y=self.fnn_percentages,
                mode="lines+markers",
                line=dict(color="#FF3131", width=2),
                marker=dict(size=6),
                name="FNN %",
            ),
            row=1,
            col=2,
        )
        fig.add_hline(
            y=5.0,
            line_dash="dot",
            line_color="#888888",
            annotation_text="5% Threshold",
            row=1,
            col=2,
        )
        fig.add_vline(
            x=self.m,
            line_dash="dash",
            line_color="#39FF14",
            annotation_text=f"m* = {self.m}",
            row=1,
            col=2,
        )

        # 2D Delay Map
        r_t = self.log_returns[self.tau :]
        r_tau = self.log_returns[: -self.tau]
        fig.add_trace(
            go.Scatter(
                x=r_tau,
                y=r_t,
                mode="markers",
                marker=dict(size=3, color=r_t, colorscale="Viridis", opacity=0.7),
                name="Delay Orbit",
            ),
            row=1,
            col=3,
        )

        fig.update_layout(
            template="plotly_dark",
            height=420,
            showlegend=False,
            margin=dict(l=30, r=30, t=60, b=30),
        )
        fig.update_xaxes(title_text="Time Delay (τ)", row=1, col=1)
        fig.update_yaxes(title_text="AMI Score (bits)", row=1, col=1)
        fig.update_xaxes(title_text="Embedding Dimension (m)", row=1, col=2)
        fig.update_yaxes(title_text="False Neighbors (%)", row=1, col=2)
        fig.update_xaxes(title_text=f"r(t - {self.tau})", row=1, col=3)
        fig.update_yaxes(title_text="r(t)", row=1, col=3)

        return fig

    def build_3d_state_space_figure(self) -> go.Figure:
        n_embed = len(self.embedded_matrix)
        tau = self.tau
        start_price_idx = (self.m - 1) * tau + 1

        aligned_dates = self.dates[start_price_idx : start_price_idx + n_embed]
        curr_prices = self.prices[start_price_idx : start_price_idx + n_embed]
        prev_prices = self.prices[start_price_idx - 1 : start_price_idx + n_embed - 1]
        curr_returns = self.log_returns[start_price_idx - 1 : start_price_idx + n_embed - 1]
        prev_returns = self.log_returns[start_price_idx - 2 : start_price_idx + n_embed - 2]

        hover_texts = []
        for d, cp, pp, cr, pr, coord in zip(
            aligned_dates, curr_prices, prev_prices, curr_returns, prev_returns, self.embedded_matrix
        ):
            cr_pct = (np.exp(cr) - 1.0) * 100.0
            pr_pct = (np.exp(pr) - 1.0) * 100.0
            txt = (
                f"<b>Node Date:</b> {d.strftime('%Y-%m-%d')}<br>"
                f"━━━━━━━━━━━━━━━━━━━━━━<br>"
                f"<b>Current Price (P_t):</b> ${cp:.2f}<br>"
                f"<b>Prev Price (P_t-1):</b> ${pp:.2f}<br>"
                f"<b>Current Return (r_t):</b> {cr_pct:+.2f}%<br>"
                f"<b>Prev Return (r_t-1):</b> {pr_pct:+.2f}%<br>"
                f"━━━━━━━━━━━━━━━━━━━━━━<br>"
                f"<b>Phase Vector Coordinates:</b><br>"
                f"x¹: {coord[0]:.4f} | x²: {coord[1]:.4f} | x³: {coord[2]:.4f}"
            )
            hover_texts.append(txt)

        fig = go.Figure()
        fig.add_trace(
            go.Scatter3d(
                x=self.embedded_matrix[:, 0],
                y=self.embedded_matrix[:, 1],
                z=self.embedded_matrix[:, 2],
                mode="lines",
                line=dict(color="rgba(120, 140, 160, 0.40)", width=1.5),
                hoverinfo="none",
                name="Phase Orbit",
            )
        )
        fig.add_trace(
            go.Scatter3d(
                x=self.embedded_matrix[:, 0],
                y=self.embedded_matrix[:, 1],
                z=self.embedded_matrix[:, 2],
                mode="markers",
                marker=dict(
                    size=3.8,
                    color=curr_returns,
                    colorscale="Viridis",
                    colorbar=dict(title="Return (r_t)", x=1.02),
                    showscale=True,
                ),
                text=hover_texts,
                hoverinfo="text",
                name="State Nodes",
            )
        )
        fig.add_trace(
            go.Scatter3d(
                x=[self.embedded_matrix[-1, 0]],
                y=[self.embedded_matrix[-1, 1]],
                z=[self.embedded_matrix[-1, 2]],
                mode="markers+text",
                marker=dict(size=9, color="#00F0FF", symbol="diamond"),
                text=[f"Current State: ${curr_prices[-1]:.2f}"],
                textposition="top center",
                name="Latest Node",
            )
        )

        fig.update_layout(
            template="plotly_dark",
            title=f"<b>3D Tangent Bundle State Space: {self.asset_name} ({self.ticker})</b><br>"
            f"Embedding Dimension m = {self.m} | Optimal Delay τ = {self.tau} days",
            height=720,
            scene=dict(
                xaxis=dict(title="x¹: r(t)", backgroundcolor="#111116"),
                yaxis=dict(title=f"x²: r(t - {tau})", backgroundcolor="#111116"),
                zaxis=dict(title=f"x³: r(t - {2*tau})", backgroundcolor="#111116"),
            ),
            margin=dict(l=10, r=10, t=60, b=10),
        )
        return fig

    def build_diagnostics_figure(self) -> go.Figure:
        fig = make_subplots(
            rows=2,
            cols=2,
            column_widths=[0.55, 0.45],
            row_heights=[0.52, 0.48],
            specs=[
                [{"type": "xy"}, {"type": "xy"}],
                [{"type": "polar"}, {"type": "table"}],
            ],
            subplot_titles=(
                "<b>Orbit Divergence & Maximum Lyapunov Exponent (LLE)</b>",
                f"<b>IAAFT Surrogate Distribution (p-value: {self.iaaft_p_value:.4f})</b>",
                "<b>Entropy Fingerprint (Pesin KS, Shannon, SampEn, PE)</b>",
                "<b>Nonlinear Dynamic Specification Table</b>",
            ),
            vertical_spacing=0.14,
            horizontal_spacing=0.08,
        )

        # LLE Divergence
        t_div, y_div = self.lyapunov_divergence_curve
        fig.add_trace(
            go.Scatter(
                x=t_div,
                y=y_div,
                mode="lines+markers",
                line=dict(color="#FF3131", width=2),
                marker=dict(size=4),
                name="Empirical <ln d(t)>",
            ),
            row=1,
            col=1,
        )

        lle = self.metrics["Max_Lyapunov_Exponent"]
        if len(t_div) > 5:
            fit_x = t_div[:8]
            fit_y = y_div[0] + lle * fit_x
            fig.add_trace(
                go.Scatter(
                    x=fit_x,
                    y=fit_y,
                    mode="lines",
                    line=dict(color="#FFFFFF", dash="dash", width=2),
                    name=f"Slope Fit λ = {lle:.4f}",
                ),
                row=1,
                col=1,
            )

        # IAAFT Histogram
        fig.add_trace(
            go.Histogram(
                x=self.iaaft_lle_distribution,
                nbinsx=12,
                marker_color="rgba(100, 180, 255, 0.65)",
                name="Surrogate LLE Distribution",
            ),
            row=1,
            col=2,
        )
        fig.add_vline(
            x=lle,
            line_width=2.5,
            line_dash="dash",
            line_color="#39FF14",
            annotation_text=f"Empirical LLE: {lle:.4f}",
            annotation_position="top right",
            row=1,
            col=2,
        )

        # Radar Plot
        r_shannon = min(self.metrics["Shannon_Entropy"] / 6.0, 1.0)
        r_samp = min(self.metrics["Sample_Entropy"] / 2.5, 1.0)
        r_perm = self.metrics["Permutation_Entropy"]
        r_pesin = min(self.metrics["Kolmogorov_Sinai_Pesin"] / 0.15, 1.0)

        radar_labels = [
            "Shannon Entropy",
            "Sample Entropy (SampEn)",
            "Permutation Entropy (PE)",
            "Kolmogorov-Sinai (Pesin)",
            "Shannon Entropy",
        ]
        radar_values = [r_shannon, r_samp, r_perm, r_pesin, r_shannon]

        fig.add_trace(
            go.Scatterpolar(
                r=radar_values,
                theta=radar_labels,
                fill="toself",
                fillcolor="rgba(255, 230, 0, 0.25)",
                line=dict(color="#FFE600", width=2),
                name="Entropy Radar",
            ),
            row=2,
            col=1,
        )

        # Specification Table
        verification_label = (
            "Deterministic Chaos Verified (H0 Rejected)"
            if self.is_chaos_verified
            else "Inconclusive / Linear Stochastic Noise"
        )

        table_headers = ["<b>Dynamic Parameter</b>", "<b>Empirical Value</b>", "<b>Physical / Statistical Interpretation</b>"]
        table_cells = [
            [
                "Embedding Delay (τ)",
                "Embedding Dimension (m)",
                "Max Lyapunov Exponent (LLE)",
                "IAAFT Significance (p-value)",
                "Pesin KS Metric Entropy (h_KS)",
                "Shannon Entropy",
                "Sample Entropy (SampEn)",
                "Permutation Entropy (PE)",
                "<b>Systemic Regime Status</b>",
            ],
            [
                f"{self.tau} days",
                f"{self.m} dimensions",
                f"{lle:.5f}",
                f"{self.iaaft_p_value:.4f}",
                f"{self.metrics['Kolmogorov_Sinai_Pesin']:.5f} nats/step",
                f"{self.metrics['Shannon_Entropy']:.3f} bits",
                f"{self.metrics['Sample_Entropy']:.4f}",
                f"{self.metrics['Permutation_Entropy']:.4f}",
                f"<b>{verification_label}</b>",
            ],
            [
                "First Minimum of AMI",
                "Phase Space Saturation (FNN)",
                "Exponential Trajectory Divergence Rate",
                "Rejection of Linear Gaussian Noise with Matching Spectrum",
                "Information Production Rate via Pesin Theorem",
                "Probability Density Uncertainty",
                "Self-Similarity & Signal Regularity",
                "Microstructure Permutation Order Diversity",
                "Finite Prediction Horizon" if self.is_chaos_verified else "Controlled Linear Regime",
            ],
        ]

        fig.add_trace(
            go.Table(
                header=dict(
                    values=table_headers,
                    fill_color="#1E1E2E",
                    font=dict(color="#FFFFFF", size=11),
                    align="center",
                ),
                cells=dict(
                    values=table_cells,
                    fill_color="#11111B",
                    font=dict(color="#E0E0E0", size=10),
                    align=["left", "center", "left"],
                    height=24,
                ),
            ),
            row=2,
            col=2,
        )

        fig.update_layout(
            template="plotly_dark",
            title=f"<b>Chaos Verification, IAAFT Surrogate Test & Pesin Entropy: {self.asset_name}</b>",
            height=850,
            showlegend=True,
            margin=dict(l=30, r=30, t=70, b=30),
        )
        fig.update_xaxes(title_text="Time Steps (Δt)", row=1, col=1)
        fig.update_yaxes(title_text="Divergence <ln d(t)>", row=1, col=1)
        fig.update_xaxes(title_text="Surrogate LLE Values", row=1, col=2)
        fig.update_yaxes(title_text="Frequency", row=1, col=2)

        return fig


# ============================================================================
# 4. Streamlit Dashboard User Interface (Clean English Layout)
# ============================================================================
def main():
    st.set_page_config(
        page_title="Financial Physics: Chaos & Entropy Engine",
        page_icon="🌌",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    st.title("🌌 Module 1: Chaos, Lyapunov Exponent & Entropy Telemetry")
    st.markdown(
        "Deterministic nonlinear dynamics pipeline: Phase space reconstruction (AMI & FNN), "
        "Largest Lyapunov Exponent (Rosenstein LLE), **IAAFT Surrogate Hypothesis Testing**, "
        "and **Kolmogorov-Sinai Metric Entropy via Pesin's Identity**."
    )

    # Sidebar Controls
    st.sidebar.header("⚙️ Market & Parameters")

    market_cat = st.sidebar.selectbox(
        "1. Select Asset Class:",
        options=list(MARKET_CATEGORIES.keys()),
        index=0,
    )

    assets_in_cat = MARKET_CATEGORIES[market_cat]
    asset_name = st.sidebar.selectbox(
        "2. Select Target Asset:",
        options=list(assets_in_cat.keys()),
        index=0,
    )
    ticker = assets_in_cat[asset_name]

    years = st.sidebar.slider(
        "3. Historical Window (Years):",
        min_value=1,
        max_value=20,
        value=10,
        step=1,
    )

    n_surrogates = st.sidebar.slider(
        "4. IAAFT Surrogate Realizations:",
        min_value=10,
        max_value=40,
        value=20,
        step=5,
    )

    st.sidebar.markdown(f"**Yahoo Finance Symbol:** `{ticker}`")
    run_btn = st.sidebar.button("🚀 Run Nonlinear Pipeline", type="primary")

    if run_btn:
        with st.spinner(f"Fetching {asset_name} data, optimizing embedding, and testing IAAFT surrogates..."):
            try:
                engine = AdvancedChaosEntropyEngine(ticker=ticker, asset_name=asset_name, years=years)
                engine.fetch_data()
                engine.reconstruct_phase_space()
                engine.compute_lyapunov_and_iaaft_test(n_surrogates=n_surrogates)
                engine.compute_entropies()

                # =============================================================
                # Summary Results Table (Matching Image Structure in English)
                # =============================================================
                st.subheader("📋 Executive Chaos & Entropy Results Summary")
                
                status_icon = "✅ Verified Chaos" if engine.is_chaos_verified else "❌ Suspicious / Inconclusive"
                lle_val = engine.metrics["Max_Lyapunov_Exponent"]
                pesin_ks = engine.metrics["Kolmogorov_Sinai_Pesin"]
                p_val = engine.iaaft_p_value
                pe_val = engine.metrics["Permutation_Entropy"]
                shannon_val = engine.metrics["Shannon_Entropy"]

                summary_df = pd.DataFrame(
                    [
                        {
                            "Deterministic Chaos Status": status_icon,
                            "Lyapunov Exponent (λ)": f"{lle_val:.4f}",
                            "Kolmogorov Entropy (Pesin)": f"{pesin_ks:.4f}",
                            "IAAFT Test (p-value)": f"{p_val:.4f}",
                            "Permutation Entropy (PE)": f"{pe_val:.4f}",
                            "Shannon Entropy (bits)": f"{shannon_val:.3f}",
                            "Embedding Delay (τ*)": f"{engine.tau}",
                            "Embedding Dimension (m*)": f"{engine.m}",
                        }
                    ]
                )
                st.dataframe(summary_df, use_container_width=True, hide_index=True)

                # =============================================================
                # Interactive Plot Tabs
                # =============================================================
                tab1, tab2, tab3 = st.tabs(
                    [
                        "📉 Takens Embedding (AMI & FNN Curves)",
                        "🪐 3D Phase Space (Price & Return Hover)",
                        "📊 Diagnostics (LLE, IAAFT & Entropies)",
                    ]
                )

                with tab1:
                    st.markdown("### 🔍 Embedding Optimization & Delay Dynamics")
                    fig_embed = engine.build_embedding_diagnostics_figure()
                    st.plotly_chart(fig_embed, use_container_width=True)

                with tab2:
                    st.markdown("### 🌐 3D Tangent Bundle (Hover on nodes to view current & previous price / return)")
                    fig_3d = engine.build_3d_state_space_figure()
                    st.plotly_chart(fig_3d, use_container_width=True)

                with tab3:
                    st.markdown("### 📈 IAAFT Surrogate Test & Pesin Metric Entropy")
                    fig_diag = engine.build_diagnostics_figure()
                    st.plotly_chart(fig_diag, use_container_width=True)

                st.success("✅ Nonlinear dynamic pipeline executed successfully.")

            except Exception as e:
                st.error(f"Execution Error: {str(e)}")
    else:
        st.info("👈 Select your desired market, asset, and historical period in the sidebar, then click **Run Nonlinear Pipeline**.")


if __name__ == "__main__":
    main()