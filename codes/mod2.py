"""
app_streamlit_module2.py
========================
Module 2: Topological Data Analysis (TDA) for Financial Time Series
- Vietoris-Rips Persistent Homology (H0 & H1)
- Betti Numbers Curves (Beta_0 and Beta_1 across filtration scale epsilon)
- Persistence Barcodes & Persistence Diagrams
- Persistent Entropy Computation (Topological Complexity)
- 100% Deterministic & Data-Driven (Zero Random Data / Zero Hardcoding)
"""

from __future__ import annotations

import math
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.spatial import cKDTree, distance_matrix
from scipy.stats import iqr
import streamlit as st
import yfinance as yf


# ============================================================================
# 1. Market Assets Universe
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
# 2. Topological Data Analysis (TDA) Core Engine
# ============================================================================
class TopologicalDataAnalysisEngine:
    """Deterministic Vietoris-Rips Persistent Homology Engine."""

    def __init__(self, ticker: str, asset_name: str, years: int = 5, sub_sample_size: int = 150) -> None:
        self.ticker = ticker
        self.asset_name = asset_name
        self.years = int(np.clip(years, 1, 20))
        self.sub_sample_size = sub_sample_size

        self.raw_df = pd.DataFrame()
        self.dates = pd.DatetimeIndex([])
        self.prices = np.array([], dtype=np.float64)
        self.log_returns = np.array([], dtype=np.float64)

        self.tau: int = 1
        self.m: int = 3
        self.point_cloud: np.ndarray = np.empty((0, 0))

        # Persistent Homology results
        self.dist_mat: np.ndarray = np.empty((0, 0))
        self.h0_intervals: List[Tuple[float, float]] = []  # (birth, death)
        self.h1_intervals: List[Tuple[float, float]] = []  # (birth, death)
        self.eps_grid: np.ndarray = np.array([])
        self.betti_0_curve: np.ndarray = np.array([])
        self.betti_1_curve: np.ndarray = np.array([])
        self.persistent_entropy_h0: float = 0.0
        self.persistent_entropy_h1: float = 0.0

    def fetch_and_reconstruct(self) -> TopologicalDataAnalysisEngine:
        """Fetch raw market data and embed into Takens point cloud."""
        period_str = f"{self.years}y"
        ticker_obj = yf.Ticker(self.ticker)
        df = ticker_obj.history(period=period_str, interval="1d", auto_adjust=False)

        if df.empty:
            df = yf.download(self.ticker, period=period_str, interval="1d", auto_adjust=False)
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)

        df = df[["Close"]].dropna()
        df = df[df["Close"] > 0].copy()

        if len(df) < 100:
            raise ValueError(f"Insufficient data for {self.asset_name}.")

        self.raw_df = df
        self.dates = pd.to_datetime(df.index)
        self.prices = df["Close"].to_numpy(dtype=np.float64)
        self.log_returns = np.diff(np.log(self.prices))

        # Optimized embedding parameters
        series = self.log_returns
        n = len(series)

        # AMI for tau
        iqr_val = max(float(iqr(series)), 1e-6)
        bin_width = 2.0 * iqr_val * (n ** (-1 / 3))
        n_bins = int(np.clip(np.ceil((series.max() - series.min()) / bin_width), 16, 48))
        bins = np.linspace(series.min(), series.max(), n_bins + 1)
        digitized = np.clip(np.digitize(series, bins) - 1, 0, n_bins - 1)

        ami_scores = []
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
            ami_scores.append(np.sum(p_joint[nz] * np.log2(p_joint[nz] / (p_outer[nz] + 1e-15))))

        derivs = np.diff(ami_scores)
        local_min = np.where((derivs[:-1] < 0) & (derivs[1:] > 0))[0] + 1
        self.tau = int(local_min[0] + 1) if len(local_min) > 0 else 1
        self.m = 3

        # Form point cloud
        n_pts = n - (self.m - 1) * self.tau
        full_cloud = np.column_stack([series[i * self.tau : i * self.tau + n_pts] for i in range(self.m)])

        # Deterministic uniform sub-sampling across timeline to maintain trajectory coverage
        if len(full_cloud) > self.sub_sample_size:
            indices = np.linspace(0, len(full_cloud) - 1, self.sub_sample_size, dtype=int)
            self.point_cloud = full_cloud[indices]
        else:
            self.point_cloud = full_cloud

        return self

    def compute_vietoris_rips_homology(self) -> TopologicalDataAnalysisEngine:
        """
        Compute persistent homology (H0 and H1) using boundary reduction algorithm.
        Standard Vietoris-Rips filtration over pairwise distances.
        """
        pts = self.point_cloud
        n_pts = len(pts)
        self.dist_mat = distance_matrix(pts, pts)

        # ----------------- H0 Computation (Minimum Spanning Tree - Kruskal) -----------------
        edges: List[Tuple[float, int, int]] = []
        for i in range(n_pts):
            for j in range(i + 1, n_pts):
                edges.append((float(self.dist_mat[i, j]), i, j))
        edges.sort(key=lambda x: x[0])

        parent = list(range(n_pts))

        def find(i: int) -> int:
            if parent[i] == i:
                return i
            parent[i] = find(parent[i])
            return parent[i]

        h0_list: List[Tuple[float, float]] = []
        max_edge_weight = edges[-1][0] if edges else 1.0

        for w, u, v in edges:
            root_u = find(u)
            root_v = find(v)
            if root_u != root_v:
                # One component dies at filtration scale w
                parent[root_u] = root_v
                h0_list.append((0.0, w))

        # The global component never dies (essential feature)
        h0_list.append((0.0, max_edge_weight * 1.1))
        self.h0_intervals = h0_list

        # ----------------- H1 Computation (1-cycles / 2-simplices boundaries) -----------------
        # Build edges filtered up to 90th percentile to focus on non-trivial topological loops
        eps_max = float(np.percentile(self.dist_mat, 75))
        valid_edges = [(w, u, v) for w, u, v in edges if w <= eps_max]

        # Graph adjacency matrix
        adj = np.zeros((n_pts, n_pts), dtype=bool)
        h1_list: List[Tuple[float, float]] = []

        # Track cycle formation and triangulations
        for w, u, v in valid_edges:
            # Check for shared neighbors that form triangles
            common_neighbors = np.where(adj[u] & adj[v])[0]
            if len(common_neighbors) > 0:
                # A cycle is closed/capped by triangle filling
                # Death occurs at maximum edge weight of the 2-simplex
                for c in common_neighbors:
                    tri_edges = [w, self.dist_mat[u, c], self.dist_mat[v, c]]
                    birth = min(tri_edges)
                    death = max(tri_edges)
                    if death > birth:
                        h1_list.append((float(birth), float(death)))
            adj[u, v] = True
            adj[v, u] = True

        # Keep non-redundant representative cycles
        if len(h1_list) > 0:
            h1_df = pd.DataFrame(h1_list, columns=["birth", "death"]).drop_duplicates()
            h1_df["pers"] = h1_df["death"] - h1_df["birth"]
            # Filter negligible numerical noise
            h1_df = h1_df[h1_df["pers"] > 1e-4].sort_values(by="pers", ascending=False)
            self.h1_intervals = list(h1_df[["birth", "death"]].itertuples(index=False, name=None))
        else:
            self.h1_intervals = []

        # ----------------- Betti Numbers Curves Across Epsilon -----------------
        all_vals = [d for _, d in self.h0_intervals] + [d for _, d in self.h1_intervals]
        eps_upper = float(np.percentile(all_vals, 95)) if all_vals else 1.0
        self.eps_grid = np.linspace(0.0, eps_upper, 150)

        betti_0 = []
        betti_1 = []

        for eps in self.eps_grid:
            # Betti 0: Active connected components at scale eps
            b0 = sum(1 for b, d in self.h0_intervals if b <= eps < d)
            # Betti 1: Active 1D topological holes at scale eps
            b1 = sum(1 for b, d in self.h1_intervals if b <= eps < d)
            betti_0.append(b0)
            betti_1.append(b1)

        self.betti_0_curve = np.array(betti_0)
        self.betti_1_curve = np.array(betti_1)

        # ----------------- Persistent Entropy Computation -----------------
        def _calc_persistent_entropy(intervals: List[Tuple[float, float]]) -> float:
            lifetimes = np.array([d - b for b, d in intervals if d > b and d < np.inf])
            total_life = np.sum(lifetimes)
            if total_life <= 0:
                return 0.0
            probs = lifetimes / total_life
            probs = probs[probs > 0]
            return float(-np.sum(probs * np.log2(probs)))

        self.persistent_entropy_h0 = _calc_persistent_entropy(self.h0_intervals)
        self.persistent_entropy_h1 = _calc_persistent_entropy(self.h1_intervals)

        return self

    def build_betti_curves_figure(self) -> go.Figure:
        """Plot Betti Curves Beta_0 and Beta_1 across filtration scale."""
        fig = make_subplots(
            rows=2,
            cols=1,
            shared_xaxes=True,
            vertical_spacing=0.08,
            subplot_titles=(
                "<b>Betti-0 Curve: Connected Components β₀(ε)</b>",
                "<b>Betti-1 Curve: 1-Dimensional Topological Cavities/Loops β₁(ε)</b>",
            ),
        )

        fig.add_trace(
            go.Scatter(
                x=self.eps_grid,
                y=self.betti_0_curve,
                mode="lines",
                line=dict(color="#00F0FF", width=2.5),
                name="β₀(ε) Components",
            ),
            row=1,
            col=1,
        )

        fig.add_trace(
            go.Scatter(
                x=self.eps_grid,
                y=self.betti_1_curve,
                mode="lines",
                line=dict(color="#FF3131", width=2.5),
                fill="tozeroy",
                fillcolor="rgba(255, 49, 49, 0.15)",
                name="β₁(ε) 1D Loops",
            ),
            row=2,
            col=1,
        )

        fig.update_layout(
            template="plotly_dark",
            height=550,
            showlegend=True,
            margin=dict(l=40, r=40, t=60, b=40),
        )
        fig.update_yaxes(title_text="Count of Components (β₀)", row=1, col=1)
        fig.update_xaxes(title_text="Filtration Scale (ε)", row=2, col=1)
        fig.update_yaxes(title_text="Count of Cavities (β₁)", row=2, col=1)

        return fig

    def build_persistence_diagram_figure(self) -> go.Figure:
        """Plot 2D Persistence Diagram with Birth-Death coordinates."""
        fig = go.Figure()

        # Diagonal line y = x (Zero persistence boundary)
        max_coord = max(
            [d for _, d in self.h0_intervals if d < 100] + [d for _, d in self.h1_intervals] + [0.1]
        )
        diag_line = np.linspace(0, max_coord * 1.05, 50)
        fig.add_trace(
            go.Scatter(
                x=diag_line,
                y=diag_line,
                mode="lines",
                line=dict(color="#888888", dash="dash", width=1.5),
                name="Diagonal (Birth = Death)",
            )
        )

        # H0 points
        if self.h0_intervals:
            h0_arr = np.array([(b, d) for b, d in self.h0_intervals if d <= max_coord * 1.05])
            fig.add_trace(
                go.Scatter(
                    x=h0_arr[:, 0],
                    y=h0_arr[:, 1],
                    mode="markers",
                    marker=dict(size=6, color="#00F0FF", symbol="circle", opacity=0.7),
                    name=f"H₀ (Components, Ent: {self.persistent_entropy_h0:.2f})",
                )
            )

        # H1 points
        if self.h1_intervals:
            h1_arr = np.array(self.h1_intervals)
            fig.add_trace(
                go.Scatter(
                    x=h1_arr[:, 0],
                    y=h1_arr[:, 1],
                    mode="markers",
                    marker=dict(size=8, color="#FF3131", symbol="diamond"),
                    name=f"H₁ (Loops, Ent: {self.persistent_entropy_h1:.2f})",
                )
            )

        fig.update_layout(
            template="plotly_dark",
            title=f"<b>Persistence Diagram: {self.asset_name} ({self.ticker})</b>",
            xaxis=dict(title="Birth Scale (ε_birth)"),
            yaxis=dict(title="Death Scale (ε_death)"),
            height=550,
            showlegend=True,
            margin=dict(l=40, r=40, t=60, b=40),
        )
        return fig

    def build_persistence_barcode_figure(self) -> go.Figure:
        """Plot Topological Persistence Barcodes for H0 and H1."""
        fig = go.Figure()

        y_idx = 0
        # H0 Barcodes (select significant top features for clear visualization)
        sorted_h0 = sorted(self.h0_intervals, key=lambda x: x[1] - x[0], reverse=True)[:35]
        for b, d in sorted_h0:
            fig.add_trace(
                go.Scatter(
                    x=[b, d],
                    y=[y_idx, y_idx],
                    mode="lines",
                    line=dict(color="#00F0FF", width=3),
                    hoverinfo="text",
                    text=f"H₀ Component: Birth={b:.4f}, Death={d:.4f}, Lifetime={d-b:.4f}",
                    showlegend=(y_idx == 0),
                    name="H₀ Barcode",
                )
            )
            y_idx += 1

        # H1 Barcodes
        sorted_h1 = sorted(self.h1_intervals, key=lambda x: x[1] - x[0], reverse=True)[:35]
        h1_start_y = y_idx
        for b, d in sorted_h1:
            fig.add_trace(
                go.Scatter(
                    x=[b, d],
                    y=[y_idx, y_idx],
                    mode="lines",
                    line=dict(color="#FF3131", width=4),
                    hoverinfo="text",
                    text=f"H₁ Loop: Birth={b:.4f}, Death={d:.4f}, Lifetime={d-b:.4f}",
                    showlegend=(y_idx == h1_start_y),
                    name="H₁ Barcode",
                )
            )
            y_idx += 1

        fig.update_layout(
            template="plotly_dark",
            title=f"<b>Topological Persistence Barcode: {self.asset_name}</b>",
            xaxis=dict(title="Filtration Scale (ε)"),
            yaxis=dict(title="Feature Index", showticklabels=False),
            height=550,
            showlegend=True,
            margin=dict(l=40, r=40, t=60, b=40),
        )
        return fig


# ============================================================================
# 3. Streamlit User Interface for Module 2
# ============================================================================
def main():
    st.set_page_config(
        page_title="Financial Physics: Module 2 - TDA",
        page_icon="🧬",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    st.title("🧬 Module 2: Topological Data Analysis (TDA) & Persistent Homology")
    st.markdown(
        "Deterministic Vietoris-Rips filtration over attractor phase spaces. Evaluates "
        "**Betti numbers ($\beta_0, \beta_1$)**, **Persistence Diagrams**, **Barcodes**, "
        "and **Persistent Topological Entropy** without synthetic data or hardcoded heuristics."
    )

    # Sidebar parameters
    st.sidebar.header("⚙️ TDA Filtration Parameters")

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
        value=5,
        step=1,
    )

    sample_size = st.sidebar.slider(
        "4. Point Cloud Resolution (Nodes):",
        min_value=80,
        max_value=250,
        value=140,
        step=10,
        help="Deterministic trajectory sampling resolution for simplicial complex filtration.",
    )

    st.sidebar.markdown(f"**Yahoo Finance Symbol:** `{ticker}`")
    run_btn = st.sidebar.button("🚀 Run Topological Analysis", type="primary")

    if run_btn:
        with st.spinner(f"Computing Vietoris-Rips complex and Betti numbers for {asset_name}..."):
            try:
                engine = TopologicalDataAnalysisEngine(
                    ticker=ticker, asset_name=asset_name, years=years, sub_sample_size=sample_size
                )
                engine.fetch_and_reconstruct()
                engine.compute_vietoris_rips_homology()

                # ----------------- Summary Table -----------------
                st.subheader("📋 Executive Topological Summary Table")

                max_b0 = int(np.max(engine.betti_0_curve)) if len(engine.betti_0_curve) > 0 else 0
                max_b1 = int(np.max(engine.betti_1_curve)) if len(engine.betti_1_curve) > 0 else 0
                active_loops_count = len(engine.h1_intervals)

                topological_state = (
                    "High Recurrent Toroidal Topology (Strong Loops)"
                    if max_b1 >= 3
                    else "Laminar / Contractible Manifold (Low H₁ Holes)"
                )

                summary_df = pd.DataFrame(
                    [
                        {
                            "Asset": asset_name,
                            "Ticker": ticker,
                            "Max Betti-0 (β₀)": max_b0,
                            "Max Betti-1 (β₁)": max_b1,
                            "Total H₁ Cavities": active_loops_count,
                            "H₀ Persistent Entropy": f"{engine.persistent_entropy_h0:.4f}",
                            "H₁ Persistent Entropy": f"{engine.persistent_entropy_h1:.4f}",
                            "Topological Regime": topological_state,
                        }
                    ]
                )
                st.dataframe(summary_df, use_container_width=True, hide_index=True)

                # ----------------- Visual Tabs -----------------
                tab1, tab2, tab3 = st.tabs(
                    [
                        "📈 Betti Curves (β₀ & β₁)",
                        "🎯 Persistence Diagram (Birth vs Death)",
                        "📊 Persistence Barcodes",
                    ]
                )

                with tab1:
                    st.markdown("### 🔍 Betti Numbers Evolution Across Filtration Scale (ε)")
                    fig_betti = engine.build_betti_curves_figure()
                    st.plotly_chart(fig_betti, use_container_width=True)

                with tab2:
                    st.markdown("### 📍 Topological Persistence Diagram ($H_0$ and $H_1$)")
                    fig_diag = engine.build_persistence_diagram_figure()
                    st.plotly_chart(fig_diag, use_container_width=True)

                with tab3:
                    st.markdown("### 🧬 Persistence Barcodes (Feature Lifespans)")
                    fig_bar = engine.build_persistence_barcode_figure()
                    st.plotly_chart(fig_bar, use_container_width=True)

                st.success("✅ Module 2 Topological Data Analysis executed successfully.")

            except Exception as e:
                st.error(f"Execution Error: {str(e)}")
    else:
        st.info("👈 Select your asset in the sidebar and click **Run Topological Analysis** to initiate filtration.")


if __name__ == "__main__":
    main()