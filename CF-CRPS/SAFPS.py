"""
SAFPS v0.1.2
State-Adaptive Financial Proper Score
Monotonic-Controller Correction

Goal
----
Fix the semantic failure observed in v0.1.1 without moving to the next
loss-function extension.

Compared methods
----------------
1) crps
2) static
3) prior
4) neural                  -> deliberately unconstrained ablation
5) hybrid                  -> bounded but not monotonic
6) monotonic_hybrid        -> semantically constrained controller

Key correction
--------------
The Monotonic Hybrid architecture is constructed so that, holding the
other state fixed:

    Risk State ↑  => pi_T cannot decrease
    Direction Uncertainty ↑ => pi_D cannot decrease

This is enforced by architecture, not only by a penalty.

Requirements
------------
pip install torch numpy pandas

Run
---
python SAFPS_v012.py
"""

import math
import random
from copy import deepcopy
from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================
# Configuration
# ============================================================

@dataclass
class Config:
    # Experiment
    n_seeds: int = 10
    seed_start: int = 42

    # Synthetic data
    n_samples: int = 7000
    train_size: int = 4500
    val_size: int = 1200

    # Training
    epochs: int = 60
    batch_size: int = 256
    lr: float = 3e-3

    # Weighted-CRPS integration grid
    z_min: float = -8.0
    z_max: float = 8.0
    n_grid: int = 129

    # Extra financial sensitivity budget
    lambda_fin: float = 2.0

    # Financial regions
    direction_width: float = 0.45
    downside_temp: float = 0.55
    tail_threshold: float = 2.2
    tail_temp: float = 0.35

    # Positive floor on allocation
    min_allocation: float = 0.05

    # Hybrid correction strength
    hybrid_delta: float = 0.35

    # Stay reasonably close to the financial prior
    prior_kl_weight: float = 0.02

    # Diagnostics
    low_risk_max: float = 0.33
    high_risk_min: float = 0.66
    stress_threshold: float = 0.70

    # Numerical tolerance for monotonicity diagnostics
    monotonic_tol: float = 1e-7


CFG = Config()

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# ============================================================
# Reproducibility
# ============================================================

def set_global_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


# ============================================================
# Synthetic financial data
# ============================================================

def make_synthetic_data(n: int, seed: int):
    """
    X[:,0] = directional signal
    X[:,1] = risk state in [0,1]
    X[:,2] = direction uncertainty in [0,1]

    y = normalized future return

    The data are deliberately:
    - heavy-tailed
    - heteroskedastic
    - left-skewed in high-risk states
    """

    rng = np.random.default_rng(seed)

    signal = rng.normal(size=n)
    risk_state = rng.beta(2.0, 3.0, size=n)
    dir_uncertainty = rng.beta(2.0, 2.0, size=n)

    # Direction signal weakens when direction uncertainty is high.
    mu = (
        0.8
        * np.tanh(signal)
        * (1.0 - 0.7 * dir_uncertainty)
    )

    # Conditional volatility grows with risk.
    sigma = 0.55 + 1.15 * risk_state

    # Heavy-tailed base innovation.
    eps = rng.standard_t(df=4, size=n) / np.sqrt(2.0)
    y = mu + sigma * eps

    # Risk-dependent asymmetric downside shocks.
    p_tail = 0.02 + 0.25 * risk_state**2

    shock = rng.random(n) < p_tail

    shock_size = (
        2.4
        + 2.1 * risk_state[shock]
        + rng.exponential(
            0.8,
            size=shock.sum()
        )
    )

    y[shock] -= shock_size

    X = np.stack(
        [
            signal,
            risk_state,
            dir_uncertainty,
        ],
        axis=1,
    ).astype(np.float32)

    return X, y.astype(np.float32)


# ============================================================
# Forecasting model
# ============================================================

class GaussianForecaster(nn.Module):
    def __init__(self, input_dim=3):
        super().__init__()

        self.backbone = nn.Sequential(
            nn.Linear(input_dim, 16),
            nn.Tanh(),
            nn.Linear(16, 16),
            nn.Tanh(),
        )

        self.mu_head = nn.Linear(16, 1)
        self.scale_head = nn.Linear(16, 1)

    def forward(self, x):
        h = self.backbone(x)

        mu = self.mu_head(h).squeeze(-1)

        sigma = (
            F.softplus(
                self.scale_head(h).squeeze(-1)
            )
            + 0.08
        )

        return mu, sigma


# ============================================================
# Gaussian CRPS
# ============================================================

def gaussian_crps(mu, sigma, y):
    """
    Closed-form Gaussian CRPS.
    Lower is better.
    """

    z = (y - mu) / sigma

    phi = (
        torch.exp(-0.5 * z**2)
        / math.sqrt(2.0 * math.pi)
    )

    Phi = 0.5 * (
        1.0
        + torch.erf(
            z / math.sqrt(2.0)
        )
    )

    return sigma * (
        z * (2.0 * Phi - 1.0)
        + 2.0 * phi
        - 1.0 / math.sqrt(math.pi)
    )


# ============================================================
# Financial regions
# ============================================================

def make_financial_regions(grid, cfg: Config):
    """
    Hierarchical soft regions:
        q_D = near zero / direction boundary
        q_N = ordinary downside
        q_T = extreme downside tail
    """

    q_tail = torch.sigmoid(
        (-grid - cfg.tail_threshold)
        / cfg.tail_temp
    )

    q_direction = (
        (1.0 - q_tail)
        * torch.exp(
            -0.5
            * (grid / cfg.direction_width) ** 2
        )
    )

    q_downside = (
        (1.0 - q_tail)
        * (1.0 - q_direction)
        * torch.sigmoid(
            (-grid - 0.35)
            / cfg.downside_temp
        )
    )

    return (
        q_direction,
        q_downside,
        q_tail,
    )


# ============================================================
# Simplex helper
# ============================================================

def bounded_simplex(logits, min_allocation: float):
    """
    Softmax + positive floor.

    For three components:
        pi_i >= min_allocation
        sum_i pi_i = 1

    The affine transformation preserves monotonicity of each
    softmax component.
    """

    k = logits.shape[1]

    if min_allocation < 0:
        raise ValueError(
            "min_allocation must be >= 0."
        )

    if k * min_allocation >= 1.0:
        raise ValueError(
            "min_allocation is too large."
        )

    raw = torch.softmax(
        logits,
        dim=1
    )

    remaining = (
        1.0
        - k * min_allocation
    )

    return (
        min_allocation
        + remaining * raw
    )


# ============================================================
# Semantically monotonic financial prior
# ============================================================

def financial_prior_logits(x):
    """
    This prior is intentionally constructed to have monotonic
    semantics.

    Holding direction uncertainty fixed:

        d g_D / d risk = -0.60
        d g_N / d risk = +0.20
        d g_T / d risk = 0.50 + 3.60*risk

    Therefore g_T has the largest risk derivative for all
    risk in [0,1], so pi_T increases with risk.

    Holding risk fixed:

        d g_D / d dir_unc = +1.40
        d g_N / d dir_unc = 0
        d g_T / d dir_unc = 0

    Therefore pi_D increases with direction uncertainty.
    """

    risk = x[:, 1]
    direction_uncertainty = x[:, 2]

    g_d = (
        0.20
        + 1.40 * direction_uncertainty
        - 0.60 * risk
    )

    g_n = (
        0.30
        + 0.20 * risk
    )

    g_t = (
        -0.20
        + 0.50 * risk
        + 1.80 * risk**2
    )

    return torch.stack(
        [g_d, g_n, g_t],
        dim=1,
    )


def financial_prior(x):
    return bounded_simplex(
        financial_prior_logits(x),
        CFG.min_allocation,
    )


# ============================================================
# Unconstrained controllers
# ============================================================

class NeuralController(nn.Module):
    """
    Intentionally unconstrained neural controller.
    It remains in the experiment as an ablation.
    """

    def __init__(self):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(2, 8),
            nn.Tanh(),
            nn.Linear(8, 3),
        )

    def forward(self, x):
        logits = self.net(
            x[:, 1:3]
        )

        return bounded_simplex(
            logits,
            CFG.min_allocation,
        )


class HybridController(nn.Module):
    """
    Financial prior + bounded but unconstrained neural correction.

    This controller can still violate the desired semantic
    monotonicity. We keep it as an ablation.
    """

    def __init__(self, delta=0.35):
        super().__init__()

        self.delta = delta

        self.net = nn.Sequential(
            nn.Linear(2, 8),
            nn.Tanh(),
            nn.Linear(8, 3),
        )

    def forward(self, x):
        prior_logits = financial_prior_logits(x)

        correction = (
            self.delta
            * torch.tanh(
                self.net(
                    x[:, 1:3]
                )
            )
        )

        final_logits = (
            prior_logits
            + correction
        )

        return bounded_simplex(
            final_logits,
            CFG.min_allocation,
        )


# ============================================================
# Monotonic neural block
# ============================================================

class Monotonic1D(nn.Module):
    """
    One-dimensional monotonic increasing neural function.

    Positive weights are enforced with softplus.

    For x in [0,1]:

        h(x) = sum_j positive_w2_j *
               softplus(positive_w1_j*x + b1_j)
               + b2

    Therefore dh/dx >= 0.

    tanh(h(x)) is also monotonic increasing.
    """

    def __init__(self, hidden=6):
        super().__init__()

        # Negative initialization -> small positive softplus weights.
        self.raw_w1 = nn.Parameter(
            torch.full(
                (hidden,),
                -1.5
            )
        )

        self.b1 = nn.Parameter(
            torch.zeros(hidden)
        )

        self.raw_w2 = nn.Parameter(
            torch.full(
                (hidden,),
                -1.5
            )
        )

        self.b2 = nn.Parameter(
            torch.zeros(1)
        )

    def forward(self, x):
        # x: [batch]

        w1 = F.softplus(
            self.raw_w1
        )

        w2 = F.softplus(
            self.raw_w2
        )

        hidden = F.softplus(
            x[:, None] * w1[None, :]
            + self.b1[None, :]
        )

        out = (
            hidden
            * w2[None, :]
        ).sum(
            dim=1
        )

        return out + self.b2


# ============================================================
# Monotonic Hybrid Controller
# ============================================================

class MonotonicHybridController(nn.Module):
    """
    Semantically constrained Hybrid Controller.

    Direction correction depends ONLY on direction uncertainty
    through a monotonic increasing network.

    Tail correction depends ONLY on risk through a monotonic
    increasing network.

    Downside logit receives no learned state correction.

    Combined with the monotonic financial prior, this ensures:

        Risk ↑  => pi_T cannot decrease
        Dir uncertainty ↑ => pi_D cannot decrease

    up to floating-point precision.
    """

    def __init__(self, delta=0.35):
        super().__init__()

        self.delta = delta

        self.dir_correction = Monotonic1D(
            hidden=6
        )

        self.tail_correction = Monotonic1D(
            hidden=6
        )

    def forward(self, x):
        base_logits = financial_prior_logits(x)

        risk = x[:, 1]
        dir_unc = x[:, 2]

        # Monotonic increasing bounded corrections.
        corr_d = (
            self.delta
            * torch.tanh(
                self.dir_correction(
                    dir_unc
                )
            )
        )

        corr_t = (
            self.delta
            * torch.tanh(
                self.tail_correction(
                    risk
                )
            )
        )

        correction = torch.stack(
            [
                corr_d,
                torch.zeros_like(corr_d),
                corr_t,
            ],
            dim=1,
        )

        final_logits = (
            base_logits
            + correction
        )

        return bounded_simplex(
            final_logits,
            CFG.min_allocation,
        )


# ============================================================
# Static allocation
# ============================================================

def static_allocation(x):
    base = torch.tensor(
        [0.35, 0.35, 0.30],
        dtype=x.dtype,
        device=x.device,
    )

    return base[None, :].repeat(
        len(x),
        1,
    )


# ============================================================
# Adaptive weighted CRPS
# ============================================================

def adaptive_weighted_crps(
    mu,
    sigma,
    y,
    allocation,
    grid,
    q_direction,
    q_downside,
    q_tail,
    lambda_fin,
):
    """
    w_t(z) =
        1 + lambda_fin *
        [
            pi_D q_D(z)
            + pi_N q_N(z)
            + pi_T q_T(z)
        ]
    """

    z = grid[None, :]

    forecast_cdf = 0.5 * (
        1.0
        + torch.erf(
            (z - mu[:, None])
            / (
                sigma[:, None]
                * math.sqrt(2.0)
            )
        )
    )

    observed_cdf = (
        z >= y[:, None]
    ).float()

    weight = (
        1.0
        + lambda_fin
        * (
            allocation[:, 0, None]
            * q_direction[None, :]
            + allocation[:, 1, None]
            * q_downside[None, :]
            + allocation[:, 2, None]
            * q_tail[None, :]
        )
    )

    integrand = (
        weight
        * (
            forecast_cdf
            - observed_cdf
        ) ** 2
    )

    return torch.trapz(
        integrand,
        grid,
        dim=1,
    )


# ============================================================
# Probability helper
# ============================================================

def normal_cdf(x):
    return 0.5 * (
        1.0
        + torch.erf(
            x / math.sqrt(2.0)
        )
    )


# ============================================================
# Method helpers
# ============================================================

METHODS = [
    "crps",
    "static",
    "prior",
    "neural",
    "hybrid",
    "monotonic_hybrid",
]


def build_controller(kind):
    if kind == "neural":
        return NeuralController().to(
            DEVICE
        )

    if kind == "hybrid":
        return HybridController(
            delta=CFG.hybrid_delta
        ).to(DEVICE)

    if kind == "monotonic_hybrid":
        return MonotonicHybridController(
            delta=CFG.hybrid_delta
        ).to(DEVICE)

    return None


def get_allocation(
    kind,
    controller,
    x
):
    if kind == "crps":
        return None

    if kind == "static":
        return static_allocation(x)

    if kind == "prior":
        return financial_prior(x)

    if kind in (
        "neural",
        "hybrid",
        "monotonic_hybrid",
    ):
        return controller(x)

    raise ValueError(
        f"Unknown method: {kind}"
    )


def prior_kl(pi, x):
    prior = financial_prior(x)

    return (
        pi
        * (
            torch.log(pi + 1e-8)
            - torch.log(
                prior + 1e-8
            )
        )
    ).sum(
        dim=1
    ).mean()


# ============================================================
# Validation objective
# ============================================================

@torch.no_grad()
def validation_objective(
    kind,
    model,
    controller,
    X_val,
    y_val,
    grid,
    qd,
    qn,
    qt,
):
    model.eval()

    if controller is not None:
        controller.eval()

    mu, sigma = model(
        X_val
    )

    if kind == "crps":
        return gaussian_crps(
            mu,
            sigma,
            y_val,
        ).mean().item()

    pi = get_allocation(
        kind,
        controller,
        X_val,
    )

    score = adaptive_weighted_crps(
        mu,
        sigma,
        y_val,
        pi,
        grid,
        qd,
        qn,
        qt,
        CFG.lambda_fin,
    ).mean()

    # Use the same regularized objective used in training
    # for learned hybrid controllers.
    if kind in (
        "hybrid",
        "monotonic_hybrid",
    ):
        score = (
            score
            + CFG.prior_kl_weight
            * prior_kl(
                pi,
                X_val
            )
        )

    return score.item()


# ============================================================
# Counterfactual monotonicity diagnostics
# ============================================================

@torch.no_grad()
def monotonicity_diagnostics(
    kind,
    controller,
):
    """
    Counterfactual checks.

    Test 1:
        hold direction uncertainty = 0.5
        sweep risk from 0 to 1
        check pi_T

    Test 2:
        hold risk = 0.5
        sweep direction uncertainty from 0 to 1
        check pi_D
    """

    if kind == "crps":
        return {
            "Risk monotonic violations": float("nan"),
            "Dir monotonic violations": float("nan"),
            "Delta pi_T risk 0->1": float("nan"),
            "Delta pi_D dir 0->1": float("nan"),
        }

    if controller is not None:
        controller.eval()

    grid_state = torch.linspace(
        0.0,
        1.0,
        101,
        device=DEVICE,
    )

    # --------------------------------------------------------
    # Risk sweep
    # --------------------------------------------------------

    x_risk = torch.zeros(
        (101, 3),
        device=DEVICE,
    )

    x_risk[:, 1] = grid_state
    x_risk[:, 2] = 0.5

    pi_risk = get_allocation(
        kind,
        controller,
        x_risk,
    )

    tail_curve = pi_risk[:, 2]

    tail_diff = (
        tail_curve[1:]
        - tail_curve[:-1]
    )

    risk_violations = int(
        (
            tail_diff
            < -CFG.monotonic_tol
        )
        .sum()
        .item()
    )

    # --------------------------------------------------------
    # Direction uncertainty sweep
    # --------------------------------------------------------

    x_dir = torch.zeros(
        (101, 3),
        device=DEVICE,
    )

    x_dir[:, 1] = 0.5
    x_dir[:, 2] = grid_state

    pi_dir = get_allocation(
        kind,
        controller,
        x_dir,
    )

    direction_curve = pi_dir[:, 0]

    direction_diff = (
        direction_curve[1:]
        - direction_curve[:-1]
    )

    dir_violations = int(
        (
            direction_diff
            < -CFG.monotonic_tol
        )
        .sum()
        .item()
    )

    return {
        "Risk monotonic violations":
            risk_violations,

        "Dir monotonic violations":
            dir_violations,

        "Delta pi_T risk 0->1":
            (
                tail_curve[-1]
                - tail_curve[0]
            ).item(),

        "Delta pi_D dir 0->1":
            (
                direction_curve[-1]
                - direction_curve[0]
            ).item(),
    }


# ============================================================
# Evaluation
# ============================================================

@torch.no_grad()
def evaluate(
    kind,
    model,
    controller,
    X,
    y,
    grid,
    qd,
    qn,
    qt,
):
    model.eval()

    if controller is not None:
        controller.eval()

    mu, sigma = model(X)

    # --------------------------------------------------------
    # CRPS
    # --------------------------------------------------------

    crps = gaussian_crps(
        mu,
        sigma,
        y,
    )

    # --------------------------------------------------------
    # Direction Brier
    # --------------------------------------------------------

    p_up = (
        1.0
        - normal_cdf(
            (0.0 - mu)
            / sigma
        )
    )

    y_up = (
        y > 0.0
    ).float()

    direction_brier = (
        p_up - y_up
    ) ** 2

    # --------------------------------------------------------
    # Tail Brier
    # --------------------------------------------------------

    threshold = (
        -CFG.tail_threshold
    )

    p_tail = normal_cdf(
        (threshold - mu)
        / sigma
    )

    y_tail = (
        y < threshold
    ).float()

    tail_brier = (
        p_tail - y_tail
    ) ** 2

    # --------------------------------------------------------
    # Stress subset
    # --------------------------------------------------------

    stress_mask = (
        X[:, 1]
        > CFG.stress_threshold
    )

    result = {
        "CRPS":
            crps.mean().item(),

        "Direction Brier":
            direction_brier.mean().item(),

        "Tail Brier":
            tail_brier.mean().item(),

        "Stress CRPS":
            crps[stress_mask].mean().item()
            if stress_mask.any()
            else float("nan"),

        "Stress Tail Brier":
            tail_brier[stress_mask].mean().item()
            if stress_mask.any()
            else float("nan"),

        "Mean predicted tail probability":
            p_tail.mean().item(),

        "Observed tail frequency":
            y_tail.mean().item(),

        "Stress predicted tail probability":
            p_tail[stress_mask].mean().item()
            if stress_mask.any()
            else float("nan"),

        "Stress observed tail frequency":
            y_tail[stress_mask].mean().item()
            if stress_mask.any()
            else float("nan"),
    }

    # --------------------------------------------------------
    # Allocation diagnostics
    # --------------------------------------------------------

    if kind != "crps":
        pi = get_allocation(
            kind,
            controller,
            X,
        )

        result["Mean pi_D"] = (
            pi[:, 0]
            .mean()
            .item()
        )

        result["Mean pi_N"] = (
            pi[:, 1]
            .mean()
            .item()
        )

        result["Mean pi_T"] = (
            pi[:, 2]
            .mean()
            .item()
        )

        risk_np = (
            X[:, 1]
            .detach()
            .cpu()
            .numpy()
        )

        dir_unc_np = (
            X[:, 2]
            .detach()
            .cpu()
            .numpy()
        )

        pi_np = (
            pi.detach()
            .cpu()
            .numpy()
        )

        result["Corr(risk, pi_T)"] = (
            np.corrcoef(
                risk_np,
                pi_np[:, 2]
            )[0, 1]
            if (
                np.std(risk_np) > 1e-8
                and np.std(pi_np[:, 2]) > 1e-8
            )
            else 0.0
        )

        result["Corr(dir_unc, pi_D)"] = (
            np.corrcoef(
                dir_unc_np,
                pi_np[:, 0]
            )[0, 1]
            if (
                np.std(dir_unc_np) > 1e-8
                and np.std(pi_np[:, 0]) > 1e-8
            )
            else 0.0
        )

        floor = CFG.min_allocation

        result["Near-floor pi_D rate"] = (
            (
                pi[:, 0]
                <= floor + 0.005
            )
            .float()
            .mean()
            .item()
        )

        result["Near-floor pi_N rate"] = (
            (
                pi[:, 1]
                <= floor + 0.005
            )
            .float()
            .mean()
            .item()
        )

        result["Near-floor pi_T rate"] = (
            (
                pi[:, 2]
                <= floor + 0.005
            )
            .float()
            .mean()
            .item()
        )

    result.update(
        monotonicity_diagnostics(
            kind,
            controller,
        )
    )

    return result


# ============================================================
# Allocation by risk bin
# ============================================================

@torch.no_grad()
def allocation_by_risk_bin(
    seed,
    kind,
    controller,
    X,
):
    if kind == "crps":
        return []

    if controller is not None:
        controller.eval()

    pi = get_allocation(
        kind,
        controller,
        X,
    )

    risk = X[:, 1]

    masks = {
        "Low": (
            risk
            < CFG.low_risk_max
        ),

        "Medium": (
            (risk >= CFG.low_risk_max)
            & (risk < CFG.high_risk_min)
        ),

        "High": (
            risk
            >= CFG.high_risk_min
        ),
    }

    rows = []

    for risk_bin, mask in masks.items():
        if not mask.any():
            continue

        rows.append(
            {
                "Seed": seed,
                "Method": kind,
                "Risk Bin": risk_bin,
                "Count": int(
                    mask.sum().item()
                ),
                "pi_D": (
                    pi[mask, 0]
                    .mean()
                    .item()
                ),
                "pi_N": (
                    pi[mask, 1]
                    .mean()
                    .item()
                ),
                "pi_T": (
                    pi[mask, 2]
                    .mean()
                    .item()
                ),
            }
        )

    return rows


# ============================================================
# Training
# ============================================================

def train_model(
    seed,
    kind,
    base_model_state,
    X_train,
    y_train,
    X_val,
    y_val,
    grid,
    qd,
    qn,
    qt,
):
    """
    Every method starts from the same forecaster initialization
    within each seed.
    """

    model = GaussianForecaster().to(
        DEVICE
    )

    model.load_state_dict(
        deepcopy(
            base_model_state
        )
    )

    controller_seed = (
        seed * 1000
        + {
            "crps": 0,
            "static": 1,
            "prior": 2,
            "neural": 3,
            "hybrid": 4,
            "monotonic_hybrid": 5,
        }[kind]
    )

    torch.manual_seed(
        controller_seed
    )

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(
            controller_seed
        )

    controller = build_controller(
        kind
    )

    params = list(
        model.parameters()
    )

    if controller is not None:
        params += list(
            controller.parameters()
        )

    optimizer = torch.optim.Adam(
        params,
        lr=CFG.lr,
    )

    best_state = None
    best_val = float("inf")
    best_epoch = -1

    n = len(
        X_train
    )

    for epoch in range(
        CFG.epochs
    ):
        model.train()

        if controller is not None:
            controller.train()

        # Same batch ordering across methods for a seed.
        generator = torch.Generator(
            device="cpu"
        )

        generator.manual_seed(
            seed * 100000
            + epoch
        )

        order = torch.randperm(
            n,
            generator=generator,
        )

        for start in range(
            0,
            n,
            CFG.batch_size,
        ):
            idx_cpu = order[
                start:
                start + CFG.batch_size
            ]

            idx = idx_cpu.to(
                DEVICE
            )

            xb = X_train[idx]
            yb = y_train[idx]

            mu, sigma = model(
                xb
            )

            # ------------------------------------------------
            # CRPS baseline
            # ------------------------------------------------

            if kind == "crps":
                loss = gaussian_crps(
                    mu,
                    sigma,
                    yb,
                ).mean()

            # ------------------------------------------------
            # Adaptive variants
            # ------------------------------------------------

            else:
                pi = get_allocation(
                    kind,
                    controller,
                    xb,
                )

                loss = adaptive_weighted_crps(
                    mu,
                    sigma,
                    yb,
                    pi,
                    grid,
                    qd,
                    qn,
                    qt,
                    CFG.lambda_fin,
                ).mean()

                # Learned hybrid controllers receive a prior
                # closeness regularizer.
                if kind in (
                    "hybrid",
                    "monotonic_hybrid",
                ):
                    loss = (
                        loss
                        + CFG.prior_kl_weight
                        * prior_kl(
                            pi,
                            xb,
                        )
                    )

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # ----------------------------------------------------
        # Validation checkpoint
        # ----------------------------------------------------

        val_score = validation_objective(
            kind,
            model,
            controller,
            X_val,
            y_val,
            grid,
            qd,
            qn,
            qt,
        )

        if val_score < best_val:
            best_val = val_score
            best_epoch = epoch + 1

            best_state = {
                "model": deepcopy(
                    model.state_dict()
                ),

                "controller": (
                    deepcopy(
                        controller.state_dict()
                    )
                    if controller is not None
                    else None
                ),
            }

    model.load_state_dict(
        best_state["model"]
    )

    if controller is not None:
        controller.load_state_dict(
            best_state[
                "controller"
            ]
        )

    return (
        model,
        controller,
        best_epoch,
        best_val,
    )


# ============================================================
# One seed
# ============================================================

def run_one_seed(seed: int):
    set_global_seed(seed)

    X_np, y_np = make_synthetic_data(
        CFG.n_samples,
        seed,
    )

    split_rng = np.random.default_rng(
        seed + 9999
    )

    indices = split_rng.permutation(
        CFG.n_samples
    )

    train_end = CFG.train_size

    val_end = (
        CFG.train_size
        + CFG.val_size
    )

    train_idx = indices[
        :train_end
    ]

    val_idx = indices[
        train_end:
        val_end
    ]

    test_idx = indices[
        val_end:
    ]

    X_train = torch.tensor(
        X_np[train_idx],
        device=DEVICE,
    )

    y_train = torch.tensor(
        y_np[train_idx],
        device=DEVICE,
    )

    X_val = torch.tensor(
        X_np[val_idx],
        device=DEVICE,
    )

    y_val = torch.tensor(
        y_np[val_idx],
        device=DEVICE,
    )

    X_test = torch.tensor(
        X_np[test_idx],
        device=DEVICE,
    )

    y_test = torch.tensor(
        y_np[test_idx],
        device=DEVICE,
    )

    grid = torch.linspace(
        CFG.z_min,
        CFG.z_max,
        CFG.n_grid,
        device=DEVICE,
    )

    qd, qn, qt = make_financial_regions(
        grid,
        CFG,
    )

    # Same forecast-model initialization for all methods.
    torch.manual_seed(
        seed + 12345
    )

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(
            seed + 12345
        )

    base_model = GaussianForecaster().to(
        DEVICE
    )

    base_model_state = deepcopy(
        base_model.state_dict()
    )

    metric_rows = []
    allocation_rows = []

    for kind in METHODS:
        print(
            f"  Training: {kind}"
        )

        model, controller, best_epoch, best_val = train_model(
            seed,
            kind,
            base_model_state,
            X_train,
            y_train,
            X_val,
            y_val,
            grid,
            qd,
            qn,
            qt,
        )

        metrics = evaluate(
            kind,
            model,
            controller,
            X_test,
            y_test,
            grid,
            qd,
            qn,
            qt,
        )

        metrics["Seed"] = seed
        metrics["Method"] = kind
        metrics["Best Epoch"] = best_epoch
        metrics["Best Validation Objective"] = best_val

        metric_rows.append(
            metrics
        )

        allocation_rows.extend(
            allocation_by_risk_bin(
                seed,
                kind,
                controller,
                X_test,
            )
        )

    return (
        metric_rows,
        allocation_rows,
    )


# ============================================================
# Reporting helpers
# ============================================================

def format_mean_std(
    mean_value,
    std_value,
):
    if pd.isna(mean_value):
        return "NaN"

    if pd.isna(std_value):
        return (
            f"{mean_value:.6f}"
        )

    return (
        f"{mean_value:.6f}"
        f" ± "
        f"{std_value:.6f}"
    )


def make_summary_table(
    metrics_df
):
    main_metrics = [
        "CRPS",
        "Direction Brier",
        "Tail Brier",
        "Stress CRPS",
        "Stress Tail Brier",
        "Mean predicted tail probability",
        "Observed tail frequency",
        "Stress predicted tail probability",
        "Stress observed tail frequency",
        "Mean pi_D",
        "Mean pi_N",
        "Mean pi_T",
        "Corr(risk, pi_T)",
        "Corr(dir_unc, pi_D)",
        "Near-floor pi_D rate",
        "Near-floor pi_N rate",
        "Near-floor pi_T rate",
        "Risk monotonic violations",
        "Dir monotonic violations",
        "Delta pi_T risk 0->1",
        "Delta pi_D dir 0->1",
        "Best Epoch",
    ]

    means = (
        metrics_df
        .groupby("Method")[
            main_metrics
        ]
        .mean()
    )

    stds = (
        metrics_df
        .groupby("Method")[
            main_metrics
        ]
        .std(ddof=1)
    )

    rows = {}

    for method in means.index:
        rows[method] = {}

        for metric in main_metrics:
            rows[method][metric] = (
                format_mean_std(
                    means.loc[
                        method,
                        metric,
                    ],
                    stds.loc[
                        method,
                        metric,
                    ],
                )
            )

    return pd.DataFrame.from_dict(
        rows,
        orient="index",
    )


def make_numeric_summary(
    metrics_df
):
    numeric_cols = (
        metrics_df
        .select_dtypes(
            include=[np.number]
        )
        .columns
        .tolist()
    )

    numeric_cols = [
        c
        for c in numeric_cols
        if c != "Seed"
    ]

    summary = (
        metrics_df
        .groupby("Method")[
            numeric_cols
        ]
        .agg(
            ["mean", "std"]
        )
    )

    summary.columns = [
        f"{metric}_{stat}"
        for metric, stat
        in summary.columns
    ]

    return summary.reset_index()


def paired_deltas_vs_crps(
    metrics_df
):
    metrics = [
        "CRPS",
        "Direction Brier",
        "Tail Brier",
        "Stress CRPS",
        "Stress Tail Brier",
    ]

    baseline = (
        metrics_df[
            metrics_df["Method"]
            == "crps"
        ]
        .set_index("Seed")
    )

    rows = []

    for method in [
        x
        for x in METHODS
        if x != "crps"
    ]:
        current = (
            metrics_df[
                metrics_df["Method"]
                == method
            ]
            .set_index("Seed")
        )

        common_seeds = (
            baseline.index
            .intersection(
                current.index
            )
        )

        for metric in metrics:
            delta = (
                current.loc[
                    common_seeds,
                    metric,
                ]
                - baseline.loc[
                    common_seeds,
                    metric,
                ]
            )

            base_values = baseline.loc[
                common_seeds,
                metric,
            ]

            relative_pct = (
                100.0
                * delta
                / base_values
            )

            rows.append(
                {
                    "Method":
                        method,

                    "Metric":
                        metric,

                    "Mean Delta":
                        delta.mean(),

                    "Std Delta":
                        delta.std(
                            ddof=1
                        ),

                    "Mean Relative Change %":
                        relative_pct.mean(),

                    "Seeds Better Than CRPS":
                        int(
                            (delta < 0)
                            .sum()
                        ),

                    "Total Seeds":
                        int(
                            len(delta)
                        ),
                }
            )

    return pd.DataFrame(
        rows
    )


def summarize_allocations(
    allocation_df
):
    if allocation_df.empty:
        return pd.DataFrame()

    grouped = (
        allocation_df
        .groupby(
            [
                "Method",
                "Risk Bin",
            ]
        )[
            [
                "pi_D",
                "pi_N",
                "pi_T",
            ]
        ]
        .agg(
            ["mean", "std"]
        )
    )

    grouped.columns = [
        f"{metric}_{stat}"
        for metric, stat
        in grouped.columns
    ]

    return grouped.reset_index()


# ============================================================
# Main
# ============================================================

def main():
    pd.set_option(
        "display.max_columns",
        None,
    )

    pd.set_option(
        "display.width",
        260,
    )

    pd.set_option(
        "display.max_colwidth",
        None,
    )

    print("=" * 130)
    print("SAFPS v0.1.2 — MONOTONIC CONTROLLER CORRECTION")
    print("=" * 130)
    print(f"Device: {DEVICE}")
    print(f"Seeds: {CFG.n_seeds}")
    print(f"Methods: {', '.join(METHODS)}")
    print(f"Epochs per model: {CFG.epochs}")
    print(f"Minimum allocation: {CFG.min_allocation:.3f}")
    print("=" * 130)

    all_metric_rows = []
    all_allocation_rows = []

    seeds = [
        CFG.seed_start + i
        for i in range(
            CFG.n_seeds
        )
    ]

    for i, seed in enumerate(
        seeds,
        start=1,
    ):
        print(
            f"\nSeed {i}/{CFG.n_seeds}: {seed}"
        )

        metric_rows, allocation_rows = run_one_seed(
            seed
        )

        all_metric_rows.extend(
            metric_rows
        )

        all_allocation_rows.extend(
            allocation_rows
        )

    metrics_df = pd.DataFrame(
        all_metric_rows
    )

    allocation_df = pd.DataFrame(
        all_allocation_rows
    )

    # --------------------------------------------------------
    # Save raw results
    # --------------------------------------------------------

    metrics_df.to_csv(
        "safps_v012_per_seed_metrics.csv",
        index=False,
    )

    allocation_df.to_csv(
        "safps_v012_per_seed_allocations.csv",
        index=False,
    )

    # --------------------------------------------------------
    # Main summary
    # --------------------------------------------------------

    summary_display = make_summary_table(
        metrics_df
    )

    numeric_summary = make_numeric_summary(
        metrics_df
    )

    numeric_summary.to_csv(
        "safps_v012_summary_numeric.csv",
        index=False,
    )

    key_columns = [
        "CRPS",
        "Direction Brier",
        "Tail Brier",
        "Stress CRPS",
        "Stress Tail Brier",
    ]

    print("\n" + "=" * 130)
    print("MAIN RESULTS: MEAN ± STD ACROSS SEEDS")
    print("=" * 130)

    print(
        summary_display[
            key_columns
        ].to_string()
    )

    # --------------------------------------------------------
    # Monotonicity / controller semantics
    # --------------------------------------------------------

    semantic_columns = [
        "Mean pi_D",
        "Mean pi_N",
        "Mean pi_T",
        "Corr(risk, pi_T)",
        "Corr(dir_unc, pi_D)",
        "Risk monotonic violations",
        "Dir monotonic violations",
        "Delta pi_T risk 0->1",
        "Delta pi_D dir 0->1",
        "Near-floor pi_D rate",
        "Near-floor pi_N rate",
        "Near-floor pi_T rate",
    ]

    print("\n" + "=" * 130)
    print("CONTROLLER SEMANTICS AND MONOTONICITY")
    print("=" * 130)

    print(
        summary_display[
            semantic_columns
        ].to_string()
    )

    # --------------------------------------------------------
    # Tail calibration
    # --------------------------------------------------------

    tail_columns = [
        "Mean predicted tail probability",
        "Observed tail frequency",
        "Stress predicted tail probability",
        "Stress observed tail frequency",
    ]

    print("\n" + "=" * 130)
    print("TAIL-PROBABILITY DIAGNOSTICS")
    print("=" * 130)

    print(
        summary_display[
            tail_columns
        ].to_string()
    )

    # --------------------------------------------------------
    # Paired deltas vs baseline
    # --------------------------------------------------------

    delta_df = paired_deltas_vs_crps(
        metrics_df
    )

    delta_df.to_csv(
        "safps_v012_paired_deltas_vs_crps.csv",
        index=False,
    )

    print("\n" + "=" * 130)
    print("PAIRED DELTAS VS CRPS BASELINE")
    print("Negative Mean Delta = improvement for lower-is-better metrics")
    print("=" * 130)

    print(
        delta_df
        .round(6)
        .to_string(
            index=False
        )
    )

    # --------------------------------------------------------
    # Allocation by risk bin
    # --------------------------------------------------------

    allocation_summary = summarize_allocations(
        allocation_df
    )

    allocation_summary.to_csv(
        "safps_v012_allocation_by_risk_summary.csv",
        index=False,
    )

    print("\n" + "=" * 130)
    print("ALLOCATION BY RISK BIN")
    print("=" * 130)

    if allocation_summary.empty:
        print(
            "No allocation diagnostics."
        )
    else:
        display_cols = [
            "Method",
            "Risk Bin",
            "pi_D_mean",
            "pi_N_mean",
            "pi_T_mean",
        ]

        print(
            allocation_summary[
                display_cols
            ]
            .round(6)
            .to_string(
                index=False
            )
        )

    # --------------------------------------------------------
    # Saved files
    # --------------------------------------------------------

    print("\n" + "=" * 130)
    print("SAVED FILES")
    print("=" * 130)

    print(
        "1) safps_v012_per_seed_metrics.csv"
    )

    print(
        "2) safps_v012_per_seed_allocations.csv"
    )

    print(
        "3) safps_v012_summary_numeric.csv"
    )

    print(
        "4) safps_v012_paired_deltas_vs_crps.csv"
    )

    print(
        "5) safps_v012_allocation_by_risk_summary.csv"
    )

    print("\nFinished.")


if __name__ == "__main__":
    main()
