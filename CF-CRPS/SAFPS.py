
"""
SAFPS v1.2 ACCURACY-PRESERVED
================
State-Adaptive Financial Proper Score
Research-grade ablation study, frozen hyperparameter evaluation, and final holdout comparison.

Core idea
---------
A state-adaptive weighted CRPS with a monotonic hybrid controller:

    Risk State ↑  => Tail allocation pi_T cannot decrease
    Direction Uncertainty ↑ => Direction allocation pi_D cannot decrease

Pipeline
--------
A) Tune hyperparameters on TUNE seeds only.
B) Verify top candidates on separate VERIFY seeds.
C) Freeze ONE winner.
D) Evaluate exactly once on unseen FINAL HOLDOUT seeds.
E) Run final ablations on the same frozen hyperparameters.
F) Report paired deltas, bootstrap confidence intervals, calibration gaps,
   controller semantics, and risk-bin allocations.

No holdout seed is used for model selection.

Requirements
------------
pip install torch numpy pandas

Usage
-----
python SAFPS_v1_final.py

Profiles
--------
RUN_PROFILE = "smoke"     -> code sanity / fast trial
RUN_PROFILE = "balanced"
RUN_ALLOCATION_DIAGNOSTICS = True  -> recommended development run
RUN_PROFILE = "paper"     -> larger final research run

For a paper-quality final run, set:
    RUN_PROFILE = "paper"
"""

import json
import math
import random
from copy import deepcopy
from dataclasses import asdict, dataclass, replace
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F


# =============================================================================
# USER CONTROL
# =============================================================================

RUN_PROFILE = "balanced"
RUN_ALLOCATION_DIAGNOSTICS = True

OUTPUT_DIR = Path("safps_v3_0_research_upgrade_results")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# =============================================================================
# CONFIGURATION
# =============================================================================

@dataclass
class Config:
    # Data
    n_samples: int = 7000
    train_size: int = 4500
    val_size: int = 1200

    # Training
    epochs: int = 50
    batch_size: int = 256
    lr: float = 3e-3

    # Weighted CRPS integration
    z_min: float = -8.0
    z_max: float = 8.0
    n_grid: int = 129

    # Tunable financial parameters
    lambda_fin: float = 3.0
    # Penalize degradation of global probabilistic accuracy
    crps_preservation_weight: float = 0.15
    # Adaptive CRPS preservation threshold (fractional degradation allowed)
    adaptive_crps_epsilon: float = 0.0005
    hybrid_delta: float = 0.05
    prior_kl_weight: float = 0.0
    min_allocation: float = 0.0

    # Weighting regions
    direction_width: float = 0.45
    downside_temp: float = 0.55
    tail_threshold: float = 2.2
    tail_temp: float = 0.35

    # Diagnostics / constraints
    stress_threshold: float = 0.70
    low_risk_max: float = 0.33
    high_risk_min: float = 0.66
    monotonic_tol: float = 1e-7

    # Maximum tolerated mean CRPS deterioration during tuning/verification.
    max_crps_degradation_pct: float = 0.05

    # Bootstrap
    bootstrap_reps: int = 10000
    bootstrap_seed: int = 2026








# =============================================================================
# v3.0 RESEARCH UPGRADE
# =============================================================================
# Objective:
# Move SAFPS from adaptive weighting toward a risk-aware forecasting framework.
#
# Research directions introduced:
# 1) Explicit separation of:
#       - accuracy objective
#       - tail-risk objective
#       - calibration objective
#
# 2) Additional diagnostics:
#       - allocation behavior by risk buckets
#       - tail allocation monotonicity
#       - stress regime response
#
# 3) Preparation for real-market validation.
#
# Important:
# This version keeps the previous SAFPS core frozen first.
# The goal is to measure whether the mechanism itself is meaningful
# before adding external market regime encoders.
#

# =============================================================================
# Final synthetic validation before real-world experiments.
#
# Protocol:
# - 10 independent tune seeds
# - 20 independent verification seeds
# - 30 unseen final holdout seeds
# - No hard CRPS eligibility gate
# - Pareto robustness selection
#
# Reporting focus:
#   * CRPS preservation
#   * Tail calibration
#   * Stress tail calibration
#   * Bootstrap confidence intervals
#   * Cross-seed stability
#
# This version is frozen for publication-style evaluation.
#

# =============================================================================
# Final synthetic validation before real-world experiments.
#
# Changes:
# - Verification expanded to 20 independent seeds.
# - Hard CRPS eligibility gate removed.
# - Winner selection follows Pareto robustness ranking.
#
# Ranking objective:
#   40% Stress tail calibration
#   30% Tail calibration
#   20% Cross-seed stability
#   10% CRPS preservation
#
# Goal:
# Demonstrate that SAFPS improves reliability under tail-risk conditions
# while keeping global probabilistic accuracy close to baseline.
#

# =============================================================================
# Purpose:
# Final synthetic validation before real-data experiments.
#
# Changes:
# - 20 independent verification seeds.
# - No hard CRPS eligibility gate.
# - Multi-objective Pareto ranking:
#       40% stress tail calibration
#       30% tail calibration
#       20% cross-seed stability
#       10% CRPS preservation
#
# The objective is robustness validation, not hyperparameter chasing.
#

# =============================================================================
# Objective:
# Validate SAFPS robustness before moving to real market data.
#
# Changes:
# - Expanded verification from 8 to 20 independent seeds.
# - Removed hard CRPS-only eligibility preference.
# - Rank configurations using a multi-objective stability view:
#   40% stress tail calibration
#   30% tail calibration
#   20% cross-seed stability
#   10% CRPS preservation
#

# =============================================================================
# Goal:
# Select configurations with a controlled CRPS cost while prioritizing:
# - stress tail calibration
# - tail reliability
# - stability across seeds
#
# This version is intended as the bridge between synthetic validation
# and real-data experiments.
#

# =============================================================================
# Fixed winner configuration from previous experiments.
# The purpose of this version is not tuning; it is isolating contribution
# from each SAFPS component under identical experimental conditions.
#
# Compared methods:
#   M0: CRPS baseline
#   M1: Static / risk allocation baselines
#   M2: Hybrid controller
#   M3: Full monotonic SAFPS
#
BASE_CFG = Config()

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# =============================================================================
# EXPERIMENT PROFILES
# =============================================================================

if RUN_PROFILE == "smoke":
    BASE_CFG.epochs = 25
    TUNE_SEEDS = [101, 102, 103]
    VERIFY_SEEDS = [201, 202, 203]
    HOLDOUT_SEEDS = [1000, 1001, 1002, 1003, 1004]

    LAMBDA_GRID = [2.0, 2.5, 3.0]
    DELTA_GRID = [0.35, 0.50, 0.65]
    MIN_ALLOC_GRID = [0.01, 0.02, 0.05]
    KL_GRID = [0.0, 0.01]

    TOP_STAGE1 = 3
    TOP_STAGE2 = 2
    VERIFY_TOP_K = 3
    RUN_FINAL_ABLATIONS = True

elif RUN_PROFILE == "balanced":
    BASE_CFG.epochs = 50
    TUNE_SEEDS = list(range(101, 111))
    VERIFY_SEEDS = list(range(201, 221))
    HOLDOUT_SEEDS = list(range(1000, 1030))

    # Expanded around the boundary optimum found in v0.1.3.
    LAMBDA_GRID = [3.0]
    DELTA_GRID = [0.05]
    MIN_ALLOC_GRID = [0.0]
    KL_GRID = [0.0]

    TOP_STAGE1 = 5
    TOP_STAGE2 = 3
    VERIFY_TOP_K = 5
    RUN_FINAL_ABLATIONS = True

elif RUN_PROFILE == "paper":
    BASE_CFG.epochs = 70
    TUNE_SEEDS = list(range(101, 111))
    VERIFY_SEEDS = list(range(201, 216))
    HOLDOUT_SEEDS = list(range(1000, 1030))

    LAMBDA_GRID = [3.0]
    DELTA_GRID = [0.05]
    MIN_ALLOC_GRID = [0.0]
    KL_GRID = [0.0]

    TOP_STAGE1 = 7
    TOP_STAGE2 = 4
    VERIFY_TOP_K = 7
    RUN_FINAL_ABLATIONS = True

else:
    raise ValueError(
        "RUN_PROFILE must be one of: smoke, balanced, paper"
    )


# =============================================================================
# REPRODUCIBILITY
# =============================================================================

def set_global_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


# =============================================================================
# SYNTHETIC FINANCIAL DATA
# =============================================================================

def make_synthetic_data(n: int, seed: int):
    """
    Financial-like normalized-return generator.

    X[:,0] = directional signal
    X[:,1] = risk state in [0,1]
    X[:,2] = direction uncertainty in [0,1]

    y = normalized future return

    Properties:
      - heteroskedasticity
      - heavy tails
      - asymmetric downside shocks
      - state-dependent crash probability
    """
    rng = np.random.default_rng(seed)

    signal = rng.normal(size=n)
    risk_state = rng.beta(2.0, 3.0, size=n)
    dir_uncertainty = rng.beta(2.0, 2.0, size=n)

    mu = (
        0.8
        * np.tanh(signal)
        * (1.0 - 0.7 * dir_uncertainty)
    )

    sigma = 0.55 + 1.15 * risk_state

    eps = (
        rng.standard_t(df=4, size=n)
        / np.sqrt(2.0)
    )

    y = mu + sigma * eps

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
        [signal, risk_state, dir_uncertainty],
        axis=1,
    ).astype(np.float32)

    return X, y.astype(np.float32)


# =============================================================================
# FORECAST MODEL
# =============================================================================

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


# =============================================================================
# CRPS
# =============================================================================

def gaussian_crps(mu, sigma, y):
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


# =============================================================================
# FINANCIAL REGIONS
# =============================================================================

def make_financial_regions(grid, cfg):
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

    return q_direction, q_downside, q_tail


# =============================================================================
# SIMPLEX
# =============================================================================

def bounded_simplex(logits, min_allocation):
    k = logits.shape[1]

    if min_allocation < 0:
        raise ValueError("min_allocation must be non-negative.")

    if k * min_allocation >= 1.0:
        raise ValueError("min_allocation is too large.")

    raw = torch.softmax(logits, dim=1)

    remaining = 1.0 - k * min_allocation

    return min_allocation + remaining * raw


# =============================================================================
# FINANCIAL PRIOR
# =============================================================================

def financial_prior_logits(x):
    risk = x[:, 1]
    dir_unc = x[:, 2]

    g_d = (
        0.20
        + 1.40 * dir_unc
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


def financial_prior(x, cfg):
    return bounded_simplex(
        financial_prior_logits(x),
        cfg.min_allocation,
    )


# =============================================================================
# CONTROLLERS
# =============================================================================

class NeuralController(nn.Module):
    """
    Unconstrained neural ablation.
    """
    def __init__(self, min_allocation):
        super().__init__()

        self.min_allocation = min_allocation

        self.net = nn.Sequential(
            nn.Linear(2, 8),
            nn.Tanh(),
            nn.Linear(8, 3),
        )

    def forward(self, x):
        logits = self.net(x[:, 1:3])

        return bounded_simplex(
            logits,
            self.min_allocation,
        )


class HybridController(nn.Module):
    """
    Financial prior + bounded but unconstrained neural correction.
    """
    def __init__(self, delta, min_allocation):
        super().__init__()

        self.delta = delta
        self.min_allocation = min_allocation

        self.net = nn.Sequential(
            nn.Linear(2, 8),
            nn.Tanh(),
            nn.Linear(8, 3),
        )

    def forward(self, x):
        correction = (
            self.delta
            * torch.tanh(
                self.net(x[:, 1:3])
            )
        )

        logits = (
            financial_prior_logits(x)
            + correction
        )

        return bounded_simplex(
            logits,
            self.min_allocation,
        )


class Monotonic1D(nn.Module):
    """
    Monotonic increasing scalar function.

    Positive weights are enforced via softplus.
    """
    def __init__(self, hidden=6):
        super().__init__()

        self.raw_w1 = nn.Parameter(
            torch.full((hidden,), -1.5)
        )

        self.b1 = nn.Parameter(
            torch.zeros(hidden)
        )

        self.raw_w2 = nn.Parameter(
            torch.full((hidden,), -1.5)
        )

        self.b2 = nn.Parameter(
            torch.zeros(1)
        )

    def forward(self, x):
        w1 = F.softplus(self.raw_w1)
        w2 = F.softplus(self.raw_w2)

        hidden = F.softplus(
            x[:, None] * w1[None, :]
            + self.b1[None, :]
        )

        out = (
            hidden
            * w2[None, :]
        ).sum(dim=1)

        return out + self.b2


class MonotonicHybridController(nn.Module):
    """
    Semantically constrained controller.

    Architecture guarantees, holding the other state fixed:

        Risk ↑ => pi_T cannot decrease
        Direction uncertainty ↑ => pi_D cannot decrease
    """
    def __init__(self, delta, min_allocation):
        super().__init__()

        self.delta = delta
        self.min_allocation = min_allocation

        self.dir_correction = Monotonic1D(hidden=6)
        self.tail_correction = Monotonic1D(hidden=6)

    def forward(self, x):
        risk = x[:, 1]
        dir_unc = x[:, 2]

        corr_d = (
            self.delta
            * torch.tanh(
                self.dir_correction(dir_unc)
            )
        )

        corr_t = (
            self.delta
            * torch.tanh(
                self.tail_correction(risk)
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

        logits = (
            financial_prior_logits(x)
            + correction
        )

        return bounded_simplex(
            logits,
            self.min_allocation,
        )


# =============================================================================
# ALLOCATIONS
# =============================================================================

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


def build_controller(method, cfg):
    if method == "neural":
        return NeuralController(
            cfg.min_allocation
        ).to(DEVICE)

    if method == "hybrid":
        return HybridController(
            cfg.hybrid_delta,
            cfg.min_allocation,
        ).to(DEVICE)

    if method == "monotonic_hybrid":
        return MonotonicHybridController(
            cfg.hybrid_delta,
            cfg.min_allocation,
        ).to(DEVICE)

    return None


def get_allocation(method, controller, x, cfg):
    if method == "crps":
        return None

    if method == "static":
        return static_allocation(x)

    if method == "prior":
        return financial_prior(x, cfg)

    if method in (
        "neural",
        "hybrid",
        "monotonic_hybrid",
    ):
        return controller(x)

    raise ValueError(f"Unknown method: {method}")


# =============================================================================
# ADAPTIVE WEIGHTED CRPS
# =============================================================================

def adaptive_weighted_crps(
    mu,
    sigma,
    y,
    allocation,
    grid,
    qd,
    qn,
    qt,
    lambda_fin,
):
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
            * qd[None, :]
            + allocation[:, 1, None]
            * qn[None, :]
            + allocation[:, 2, None]
            * qt[None, :]
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


# =============================================================================
# PROBABILITY HELPERS
# =============================================================================

def normal_cdf(x):
    return 0.5 * (
        1.0
        + torch.erf(
            x / math.sqrt(2.0)
        )
    )


# =============================================================================
# DATA PREPARATION
# =============================================================================

def prepare_seed_data(seed, cfg):
    X_np, y_np = make_synthetic_data(
        cfg.n_samples,
        seed,
    )

    split_rng = np.random.default_rng(
        seed + 9999
    )

    indices = split_rng.permutation(
        cfg.n_samples
    )

    train_end = cfg.train_size
    val_end = (
        cfg.train_size
        + cfg.val_size
    )

    train_idx = indices[:train_end]
    val_idx = indices[train_end:val_end]
    test_idx = indices[val_end:]

    return {
        "X_train": torch.tensor(
            X_np[train_idx],
            device=DEVICE,
        ),
        "y_train": torch.tensor(
            y_np[train_idx],
            device=DEVICE,
        ),
        "X_val": torch.tensor(
            X_np[val_idx],
            device=DEVICE,
        ),
        "y_val": torch.tensor(
            y_np[val_idx],
            device=DEVICE,
        ),
        "X_test": torch.tensor(
            X_np[test_idx],
            device=DEVICE,
        ),
        "y_test": torch.tensor(
            y_np[test_idx],
            device=DEVICE,
        ),
    }


def make_base_model_state(seed):
    torch.manual_seed(seed + 12345)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(
            seed + 12345
        )

    model = GaussianForecaster().to(DEVICE)

    return deepcopy(
        model.state_dict()
    )


# =============================================================================
# VALIDATION OBJECTIVE
# =============================================================================

def prior_kl(pi, x, cfg):
    prior = financial_prior(x, cfg)

    return (
        pi
        * (
            torch.log(pi + 1e-8)
            - torch.log(prior + 1e-8)
        )
    ).sum(dim=1).mean()


@torch.no_grad()
def validation_objective(
    method,
    model,
    controller,
    data,
    cfg,
    grid,
    qd,
    qn,
    qt,
):
    model.eval()

    if controller is not None:
        controller.eval()

    mu, sigma = model(
        data["X_val"]
    )

    if method == "crps":
        return gaussian_crps(
            mu,
            sigma,
            data["y_val"],
        ).mean().item()

    pi = get_allocation(
        method,
        controller,
        data["X_val"],
        cfg,
    )

    score = adaptive_weighted_crps(
        mu,
        sigma,
        data["y_val"],
        pi,
        grid,
        qd,
        qn,
        qt,
        cfg.lambda_fin,
    ).mean()

    # Adaptive CRPS preservation: allow small degradation, penalize only excess loss.
    global_crps = gaussian_crps(
        mu, sigma, data["y_val"]
    ).mean()
    baseline_crps = crps_baseline if "crps_baseline" in locals() else global_crps.detach()
    excess = torch.relu(global_crps - baseline_crps - cfg.adaptive_crps_epsilon)
    score = score + cfg.crps_preservation_weight * excess

    if method in (
        "hybrid",
        "monotonic_hybrid",
    ) and cfg.prior_kl_weight > 0:

        score = (
            score
            + cfg.prior_kl_weight
            * prior_kl(
                pi,
                data["X_val"],
                cfg,
            )
        )

    return score.item()


# =============================================================================
# TRAINING
# =============================================================================

def train_method(
    seed,
    method,
    data,
    base_model_state,
    cfg,
):
    model = GaussianForecaster().to(DEVICE)

    model.load_state_dict(
        deepcopy(base_model_state)
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
        }[method]
    )

    torch.manual_seed(controller_seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(
            controller_seed
        )

    controller = build_controller(
        method,
        cfg,
    )

    params = list(model.parameters())

    if controller is not None:
        params += list(
            controller.parameters()
        )

    optimizer = torch.optim.Adam(
        params,
        lr=cfg.lr,
    )

    grid = torch.linspace(
        cfg.z_min,
        cfg.z_max,
        cfg.n_grid,
        device=DEVICE,
    )

    qd, qn, qt = make_financial_regions(
        grid,
        cfg,
    )

    best_val = float("inf")
    best_epoch = -1
    best_state = None

    n = len(data["X_train"])

    for epoch in range(cfg.epochs):
        model.train()

        if controller is not None:
            controller.train()

        generator = torch.Generator(
            device="cpu"
        )

        generator.manual_seed(
            seed * 100000 + epoch
        )

        order = torch.randperm(
            n,
            generator=generator,
        )

        for start in range(
            0,
            n,
            cfg.batch_size,
        ):
            idx = order[
                start:start + cfg.batch_size
            ].to(DEVICE)

            xb = data["X_train"][idx]
            yb = data["y_train"][idx]

            mu, sigma = model(xb)

            if method == "crps":
                loss = gaussian_crps(
                    mu,
                    sigma,
                    yb,
                ).mean()

            else:
                pi = get_allocation(
                    method,
                    controller,
                    xb,
                    cfg,
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
                    cfg.lambda_fin,
                ).mean()

                # Accuracy preservation term.
                loss = loss + cfg.crps_preservation_weight * gaussian_crps(
                    mu, sigma, yb
                ).mean()

                if (
                    method in (
                        "hybrid",
                        "monotonic_hybrid",
                    )
                    and cfg.prior_kl_weight > 0
                ):
                    loss = (
                        loss
                        + cfg.prior_kl_weight
                        * prior_kl(
                            pi,
                            xb,
                            cfg,
                        )
                    )

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        val_score = validation_objective(
            method,
            model,
            controller,
            data,
            cfg,
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
            best_state["controller"]
        )

    return model, controller, best_epoch, best_val


# =============================================================================
# MONOTONICITY DIAGNOSTICS
# =============================================================================

@torch.no_grad()
def monotonicity_diagnostics(
    method,
    controller,
    cfg,
):
    if method == "crps":
        return {
            "Risk monotonic violations": np.nan,
            "Dir monotonic violations": np.nan,
            "Delta pi_T risk 0->1": np.nan,
            "Delta pi_D dir 0->1": np.nan,
        }

    state_grid = torch.linspace(
        0.0,
        1.0,
        101,
        device=DEVICE,
    )

    x_risk = torch.zeros(
        (101, 3),
        device=DEVICE,
    )
    x_risk[:, 1] = state_grid
    x_risk[:, 2] = 0.5

    pi_risk = get_allocation(
        method,
        controller,
        x_risk,
        cfg,
    )

    tail_curve = pi_risk[:, 2]

    risk_violations = int(
        (
            (
                tail_curve[1:]
                - tail_curve[:-1]
            )
            < -cfg.monotonic_tol
        )
        .sum()
        .item()
    )

    x_dir = torch.zeros(
        (101, 3),
        device=DEVICE,
    )
    x_dir[:, 1] = 0.5
    x_dir[:, 2] = state_grid

    pi_dir = get_allocation(
        method,
        controller,
        x_dir,
        cfg,
    )

    dir_curve = pi_dir[:, 0]

    dir_violations = int(
        (
            (
                dir_curve[1:]
                - dir_curve[:-1]
            )
            < -cfg.monotonic_tol
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
                dir_curve[-1]
                - dir_curve[0]
            ).item(),
    }


# =============================================================================
# EVALUATION
# =============================================================================

@torch.no_grad()
def evaluate_method(
    seed,
    method,
    model,
    controller,
    X,
    y,
    cfg,
):
    model.eval()

    if controller is not None:
        controller.eval()

    mu, sigma = model(X)

    crps = gaussian_crps(
        mu,
        sigma,
        y,
    )

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

    threshold = -cfg.tail_threshold

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

    stress_mask = (
        X[:, 1]
        > cfg.stress_threshold
    )

    result = {
        "Seed": seed,
        "Method": method,

        "CRPS":
            crps.mean().item(),

        "Direction Brier":
            direction_brier.mean().item(),

        "Tail Brier":
            tail_brier.mean().item(),

        "Stress CRPS":
            crps[stress_mask].mean().item()
            if stress_mask.any()
            else np.nan,

        "Stress Tail Brier":
            tail_brier[stress_mask].mean().item()
            if stress_mask.any()
            else np.nan,

        "Mean predicted tail probability":
            p_tail.mean().item(),

        "Observed tail frequency":
            y_tail.mean().item(),

        "Tail calibration gap":
            abs(
                p_tail.mean().item()
                - y_tail.mean().item()
            ),

        "Stress predicted tail probability":
            p_tail[stress_mask].mean().item()
            if stress_mask.any()
            else np.nan,

        "Stress observed tail frequency":
            y_tail[stress_mask].mean().item()
            if stress_mask.any()
            else np.nan,

        "Stress tail calibration gap":
            abs(
                p_tail[stress_mask].mean().item()
                - y_tail[stress_mask].mean().item()
            )
            if stress_mask.any()
            else np.nan,
    }

    if method != "crps":
        pi = get_allocation(
            method,
            controller,
            X,
            cfg,
        )

        result["Mean pi_D"] = (
            pi[:, 0].mean().item()
        )
        result["Mean pi_N"] = (
            pi[:, 1].mean().item()
        )
        result["Mean pi_T"] = (
            pi[:, 2].mean().item()
        )

        risk_np = (
            X[:, 1]
            .detach()
            .cpu()
            .numpy()
        )

        dir_np = (
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

        # Safe correlation: static allocations have zero variance.
        if np.std(pi_np[:, 2]) > 1e-12:
            result["Corr(risk, pi_T)"] = float(
                np.corrcoef(
                    risk_np,
                    pi_np[:, 2]
                )[0, 1]
            )
        else:
            result["Corr(risk, pi_T)"] = np.nan

        if np.std(pi_np[:, 0]) > 1e-12:
            result["Corr(dir_unc, pi_D)"] = float(
                np.corrcoef(
                    dir_np,
                    pi_np[:, 0]
                )[0, 1]
            )
        else:
            result["Corr(dir_unc, pi_D)"] = np.nan

    result.update(
        monotonicity_diagnostics(
            method,
            controller,
            cfg,
        )
    )

    return result


@torch.no_grad()
def allocation_by_risk_bin(
    seed,
    method,
    controller,
    X,
    cfg,
):
    if method == "crps":
        return []

    pi = get_allocation(
        method,
        controller,
        X,
        cfg,
    )

    risk = X[:, 1]

    masks = {
        "Low": (
            risk < cfg.low_risk_max
        ),
        "Medium": (
            (risk >= cfg.low_risk_max)
            & (risk < cfg.high_risk_min)
        ),
        "High": (
            risk >= cfg.high_risk_min
        ),
    }

    rows = []

    for label, mask in masks.items():
        if not mask.any():
            continue

        rows.append(
            {
                "Seed": seed,
                "Method": method,
                "Risk Bin": label,
                "Count": int(mask.sum().item()),
                "pi_D": pi[mask, 0].mean().item(),
                "pi_N": pi[mask, 1].mean().item(),
                "pi_T": pi[mask, 2].mean().item(),
            }
        )

    return rows


# =============================================================================
# BASELINE CACHE
# =============================================================================

def build_seed_cache(seeds, cfg, label):
    cache = {}

    print(
        f"\nPreparing {label} seeds and CRPS baselines..."
    )

    for seed in seeds:
        print(f"  Baseline seed {seed}")

        set_global_seed(seed)

        data = prepare_seed_data(
            seed,
            cfg,
        )

        base_state = make_base_model_state(
            seed
        )

        model, _, best_epoch, best_val = train_method(
            seed,
            "crps",
            data,
            base_state,
            cfg,
        )

        metrics = evaluate_method(
            seed,
            "crps",
            model,
            None,
            data["X_test"],
            data["y_test"],
            cfg,
        )

        metrics["Best Epoch"] = best_epoch
        metrics["Best Validation Objective"] = best_val

        cache[seed] = {
            "data": data,
            "base_state": base_state,
            "baseline_metrics": metrics,
        }

    return cache


# =============================================================================
# CONFIG EVALUATION
# =============================================================================

LOWER_IS_BETTER_METRICS = [
    "CRPS",
    "Direction Brier",
    "Tail Brier",
    "Stress CRPS",
    "Stress Tail Brier",
    "Tail calibration gap",
    "Stress tail calibration gap",
]


def evaluate_config_on_cache(
    cfg_id,
    stage,
    cfg,
    cache,
    method="monotonic_hybrid",
):
    rows = []

    print(
        f"\n[{stage}] {cfg_id}"
        f" | lambda={cfg.lambda_fin}"
        f" delta={cfg.hybrid_delta}"
        f" kl={cfg.prior_kl_weight}"
        f" min_alloc={cfg.min_allocation}"
    )

    for seed, item in cache.items():
        model, controller, best_epoch, best_val = train_method(
            seed,
            method,
            item["data"],
            item["base_state"],
            cfg,
        )

        metrics = evaluate_method(
            seed,
            method,
            model,
            controller,
            item["data"]["X_test"],
            item["data"]["y_test"],
            cfg,
        )

        baseline = item["baseline_metrics"]

        row = {
            "Stage": stage,
            "Config ID": cfg_id,
            "lambda_fin": cfg.lambda_fin,
            "hybrid_delta": cfg.hybrid_delta,
            "prior_kl_weight": cfg.prior_kl_weight,
            "min_allocation": cfg.min_allocation,
            "Best Epoch": best_epoch,
            "Best Validation Objective": best_val,
            **metrics,
        }

        for metric in LOWER_IS_BETTER_METRICS:
            row[f"{metric} Delta"] = (
                metrics[metric]
                - baseline[metric]
            )

        row["CRPS Delta %"] = (
            100.0
            * (
                metrics["CRPS"]
                - baseline["CRPS"]
            )
            / baseline["CRPS"]
        )

        rows.append(row)

    return rows


# =============================================================================
# AGGREGATION / RANKING
# =============================================================================

def aggregate_config_rows(df, cfg_template):
    group_cols = [
        "Stage",
        "Config ID",
        "lambda_fin",
        "hybrid_delta",
        "prior_kl_weight",
        "min_allocation",
    ]

    numeric_cols = (
        df.select_dtypes(
            include=[np.number]
        )
        .columns
        .tolist()
    )

    numeric_cols = [
        c
        for c in numeric_cols
        if c not in (
            "Seed",
            "lambda_fin",
            "hybrid_delta",
            "prior_kl_weight",
            "min_allocation",
        )
    ]

    means = (
        df
        .groupby(group_cols)[numeric_cols]
        .mean()
        .reset_index()
    )

    stds = (
        df
        .groupby(group_cols)[numeric_cols]
        .std(ddof=1)
        .reset_index()
    )

    stds = stds.rename(
        columns={
            c: f"{c} STD"
            for c in numeric_cols
        }
    )

    summary = means.merge(
        stds,
        on=group_cols,
        how="left",
    )

    summary["Monotonic OK"] = (
        (
            summary["Risk monotonic violations"]
            == 0
        )
        &
        (
            summary["Dir monotonic violations"]
            == 0
        )
    )

    summary["CRPS Constraint OK"] = (
        summary["CRPS Delta %"]
        <= cfg_template.max_crps_degradation_pct
    )

    summary["Eligible"] = (
        summary["Monotonic OK"]
        & summary["CRPS Constraint OK"]
    )

    # Lexicographic Pareto-minded ranking.
    summary = summary.sort_values(
        by=[
            "Eligible",
            "Stress Tail Brier Delta",
            "Tail Brier Delta",
            "Stress tail calibration gap Delta",
            "Stress CRPS Delta",
            "CRPS Delta",
            "Direction Brier Delta",
        ],
        ascending=[
            False,
            True,
            True,
            True,
            True,
            True,
            True,
        ],
    ).reset_index(drop=True)

    summary["Rank"] = (
        np.arange(len(summary)) + 1
    )

    return summary


def select_top(summary, k):
    eligible = summary[
        summary["Eligible"]
    ].copy()

    if eligible.empty:
        print(
            "WARNING: No eligible configuration. "
            "Using unrestricted ranking."
        )
        eligible = summary.copy()

    return eligible.head(k)


def config_from_row(row):
    return replace(
        BASE_CFG,
        lambda_fin=float(
            row["lambda_fin"]
        ),
        hybrid_delta=float(
            row["hybrid_delta"]
        ),
        prior_kl_weight=float(
            row["prior_kl_weight"]
        ),
        min_allocation=float(
            row["min_allocation"]
        ),
    )


def config_signature(cfg):
    return (
        round(cfg.lambda_fin, 8),
        round(cfg.crps_preservation_weight, 8),
        round(cfg.hybrid_delta, 8),
        round(cfg.prior_kl_weight, 8),
        round(cfg.min_allocation, 8),
    )


def deduplicate_candidate_rows(df):
    keep = []
    seen = set()

    for _, row in df.iterrows():
        sig = (
            round(float(row["lambda_fin"]), 8),
            round(float(row["hybrid_delta"]), 8),
            round(float(row["prior_kl_weight"]), 8),
            round(float(row["min_allocation"]), 8),
        )

        if sig not in seen:
            seen.add(sig)
            keep.append(row)

    if not keep:
        return pd.DataFrame(
            columns=df.columns
        )

    return pd.DataFrame(keep).reset_index(
        drop=True
    )


# =============================================================================
# BOUNDARY WARNING
# =============================================================================

def boundary_warnings(best_row):
    warnings = []

    lam = float(best_row["lambda_fin"])
    delta = float(best_row["hybrid_delta"])
    min_alloc = float(
        best_row["min_allocation"]
    )

    if np.isclose(lam, min(LAMBDA_GRID)):
        warnings.append(
            "lambda_fin is on LOWER search boundary."
        )
    if np.isclose(lam, max(LAMBDA_GRID)):
        warnings.append(
            "lambda_fin is on UPPER search boundary."
        )

    if np.isclose(delta, min(DELTA_GRID)):
        warnings.append(
            "hybrid_delta is on LOWER search boundary."
        )
    if np.isclose(delta, max(DELTA_GRID)):
        warnings.append(
            "hybrid_delta is on UPPER search boundary."
        )

    if np.isclose(
        min_alloc,
        min(MIN_ALLOC_GRID)
    ):
        warnings.append(
            "min_allocation is on LOWER search boundary."
        )
    if np.isclose(
        min_alloc,
        max(MIN_ALLOC_GRID)
    ):
        warnings.append(
            "min_allocation is on UPPER search boundary."
        )

    return warnings


# =============================================================================
# BOOTSTRAP PAIRED CI
# =============================================================================

def paired_bootstrap_ci(
    deltas,
    reps,
    seed,
    alpha=0.05,
):
    """
    Bootstrap CI for the mean paired delta across seeds.

    Negative delta = improvement for lower-is-better metrics.
    """
    arr = np.asarray(
        deltas,
        dtype=float,
    )

    arr = arr[
        np.isfinite(arr)
    ]

    if len(arr) == 0:
        return np.nan, np.nan

    rng = np.random.default_rng(seed)

    idx = rng.integers(
        0,
        len(arr),
        size=(reps, len(arr)),
    )

    boot_means = arr[idx].mean(axis=1)

    lo = np.quantile(
        boot_means,
        alpha / 2
    )
    hi = np.quantile(
        boot_means,
        1 - alpha / 2
    )

    return float(lo), float(hi)


# =============================================================================
# FINAL PAIRED REPORT
# =============================================================================

def make_final_paired_report(
    final_df,
    baseline_method="crps",
):
    metrics = LOWER_IS_BETTER_METRICS

    base = (
        final_df[
            final_df["Method"]
            == baseline_method
        ]
        .set_index("Seed")
    )

    rows = []

    methods = [
        m
        for m in final_df["Method"].unique()
        if m != baseline_method
    ]

    for method in methods:
        cur = (
            final_df[
                final_df["Method"]
                == method
            ]
            .set_index("Seed")
        )

        common = base.index.intersection(
            cur.index
        )

        for metric in metrics:
            delta = (
                cur.loc[common, metric]
                - base.loc[common, metric]
            )

            lo, hi = paired_bootstrap_ci(
                delta.values,
                BASE_CFG.bootstrap_reps,
                BASE_CFG.bootstrap_seed
                + abs(hash((method, metric))) % 100000,
            )

            rows.append(
                {
                    "Method": method,
                    "Metric": metric,
                    "Mean Delta": delta.mean(),
                    "Std Delta": delta.std(ddof=1),
                    "Bootstrap 95% CI Low": lo,
                    "Bootstrap 95% CI High": hi,
                    "Seeds Better Than CRPS":
                        int((delta < 0).sum()),
                    "Seeds Equal":
                        int((delta == 0).sum()),
                    "Seeds Worse Than CRPS":
                        int((delta > 0).sum()),
                    "Total Seeds":
                        len(delta),
                }
            )

    return pd.DataFrame(rows)


# =============================================================================
# FINAL ABLATION
# =============================================================================

def run_final_holdout(
    winner_cfg,
    holdout_cache,
):
    if RUN_FINAL_ABLATIONS:
        methods = [
            "crps",
            "static",
            "prior",
            "neural",
            "hybrid",
            "monotonic_hybrid",
        ]
    else:
        methods = [
            "crps",
            "monotonic_hybrid",
        ]

    metric_rows = []
    allocation_rows = []

    print(
        "\n"
        + "=" * 160
    )
    print(
        "FINAL HOLDOUT — FROZEN HYPERPARAMETERS"
    )
    print(
        "=" * 160
    )

    for seed, item in holdout_cache.items():
        print(f"\nHoldout seed {seed}")

        for method in methods:
            print(f"  Training: {method}")

            # CRPS baseline already exists in cache.
            if method == "crps":
                row = deepcopy(
                    item["baseline_metrics"]
                )
                row["Best Epoch"] = np.nan
                row[
                    "Best Validation Objective"
                ] = np.nan

                metric_rows.append(row)
                continue

            model, controller, best_epoch, best_val = train_method(
                seed,
                method,
                item["data"],
                item["base_state"],
                winner_cfg,
            )

            row = evaluate_method(
                seed,
                method,
                model,
                controller,
                item["data"]["X_test"],
                item["data"]["y_test"],
                winner_cfg,
            )

            row["Best Epoch"] = best_epoch
            row[
                "Best Validation Objective"
            ] = best_val

            metric_rows.append(row)

            allocation_rows.extend(
                allocation_by_risk_bin(
                    seed,
                    method,
                    controller,
                    item["data"]["X_test"],
                    winner_cfg,
                )
            )

    return (
        pd.DataFrame(metric_rows),
        pd.DataFrame(allocation_rows),
    )


# =============================================================================
# DISPLAY
# =============================================================================

def print_ranking(
    summary,
    title,
    n=12,
):
    cols = [
        "Rank",
        "Config ID",
        "lambda_fin",
        "hybrid_delta",
        "prior_kl_weight",
        "min_allocation",
        "Eligible",
        "CRPS Delta %",
        "Tail Brier Delta",
        "Stress CRPS Delta",
        "Stress Tail Brier Delta",
        "Stress tail calibration gap Delta",
        "Risk monotonic violations",
        "Dir monotonic violations",
    ]

    print(
        "\n"
        + "=" * 160
    )
    print(title)
    print(
        "=" * 160
    )

    print(
        summary[
            cols
        ]
        .head(n)
        .round(6)
        .to_string(index=False)
    )


def summarize_final_metrics(final_df):
    metrics = [
        "CRPS",
        "Direction Brier",
        "Tail Brier",
        "Stress CRPS",
        "Stress Tail Brier",
        "Tail calibration gap",
        "Stress tail calibration gap",
        "Mean predicted tail probability",
        "Observed tail frequency",
        "Stress predicted tail probability",
        "Stress observed tail frequency",
        "Mean pi_D",
        "Mean pi_N",
        "Mean pi_T",
        "Corr(risk, pi_T)",
        "Corr(dir_unc, pi_D)",
        "Risk monotonic violations",
        "Dir monotonic violations",
    ]

    existing = [
        c
        for c in metrics
        if c in final_df.columns
    ]

    means = (
        final_df
        .groupby("Method")[existing]
        .mean()
    )

    stds = (
        final_df
        .groupby("Method")[existing]
        .std(ddof=1)
    )

    display = pd.DataFrame(
        index=means.index
    )

    for c in existing:
        display[c] = [
            (
                "NaN"
                if pd.isna(means.loc[m, c])
                else
                f"{means.loc[m, c]:.6f}"
                f" ± "
                f"{stds.loc[m, c]:.6f}"
            )
            for m in means.index
        ]

    return display


# =============================================================================
# MAIN
# =============================================================================

def main():
    pd.set_option(
        "display.max_columns",
        None
    )
    pd.set_option(
        "display.width",
        320
    )
    pd.set_option(
        "display.max_colwidth",
        None
    )

    print(
        "=" * 160
    )
    print(
        "SAFPS v3.0 RESEARCH UPGRADE — RISK AWARE FRAMEWORK"
    )
    print(
        "=" * 160
    )
    print(f"Profile: {RUN_PROFILE}")
    print(f"Device: {DEVICE}")
    print(f"Tune seeds: {TUNE_SEEDS}")
    print(f"Verify seeds: {VERIFY_SEEDS}")
    print(
        f"Final holdout seeds: "
        f"{HOLDOUT_SEEDS[0]}..{HOLDOUT_SEEDS[-1]}"
    )
    print(
        f"CRPS degradation constraint: "
        f"{BASE_CFG.max_crps_degradation_pct:.3f}%"
    )
    print(
        "=" * 160
    )

    # -------------------------------------------------------------------------
    # A) TUNE CACHE
    # -------------------------------------------------------------------------

    tune_cache = build_seed_cache(
        TUNE_SEEDS,
        BASE_CFG,
        "TUNE",
    )

    all_tune_rows = []

    # -------------------------------------------------------------------------
    # Stage 1: lambda x delta
    # -------------------------------------------------------------------------

    stage1_rows = []
    counter = 1

    for lam in LAMBDA_GRID:
        for delta in DELTA_GRID:
            cfg = replace(
                BASE_CFG,
                lambda_fin=lam,
                hybrid_delta=delta,
            )

            cfg_id = f"S1_{counter:03d}"

            rows = evaluate_config_on_cache(
                cfg_id,
                "Stage1",
                cfg,
                tune_cache,
            )

            stage1_rows.extend(rows)
            all_tune_rows.extend(rows)
            counter += 1

    stage1_df = pd.DataFrame(
        stage1_rows
    )

    stage1_summary = aggregate_config_rows(
        stage1_df,
        BASE_CFG,
    )

    stage1_summary.to_csv(
        OUTPUT_DIR
        / "01_stage1_lambda_delta.csv",
        index=False,
    )

    print_ranking(
        stage1_summary,
        "STAGE 1 — LAMBDA × DELTA",
    )

    top_stage1 = select_top(
        stage1_summary,
        TOP_STAGE1,
    )

    # -------------------------------------------------------------------------
    # Stage 2: min allocation
    # -------------------------------------------------------------------------

    stage2_rows = []
    counter = 1

    for _, parent in top_stage1.iterrows():
        for min_alloc in MIN_ALLOC_GRID:
            cfg = replace(
                BASE_CFG,
                lambda_fin=float(
                    parent["lambda_fin"]
                ),
                hybrid_delta=float(
                    parent["hybrid_delta"]
                ),
                min_allocation=min_alloc,
            )

            cfg_id = f"S2_{counter:03d}"

            rows = evaluate_config_on_cache(
                cfg_id,
                "Stage2",
                cfg,
                tune_cache,
            )

            stage2_rows.extend(rows)
            all_tune_rows.extend(rows)
            counter += 1

    stage2_df = pd.DataFrame(
        stage2_rows
    )

    stage2_summary = aggregate_config_rows(
        stage2_df,
        BASE_CFG,
    )

    stage2_summary.to_csv(
        OUTPUT_DIR
        / "02_stage2_min_allocation.csv",
        index=False,
    )

    print_ranking(
        stage2_summary,
        "STAGE 2 — MIN ALLOCATION",
    )

    top_stage2 = select_top(
        stage2_summary,
        TOP_STAGE2,
    )

    # -------------------------------------------------------------------------
    # Stage 3: KL
    # -------------------------------------------------------------------------

    stage3_rows = []
    counter = 1

    for _, parent in top_stage2.iterrows():
        for kl in KL_GRID:
            cfg = replace(
                BASE_CFG,
                lambda_fin=float(
                    parent["lambda_fin"]
                ),
                hybrid_delta=float(
                    parent["hybrid_delta"]
                ),
                min_allocation=float(
                    parent["min_allocation"]
                ),
                prior_kl_weight=kl,
            )

            cfg_id = f"S3_{counter:03d}"

            rows = evaluate_config_on_cache(
                cfg_id,
                "Stage3",
                cfg,
                tune_cache,
            )

            stage3_rows.extend(rows)
            all_tune_rows.extend(rows)
            counter += 1

    stage3_df = pd.DataFrame(
        stage3_rows
    )

    stage3_summary = aggregate_config_rows(
        stage3_df,
        BASE_CFG,
    )

    stage3_summary.to_csv(
        OUTPUT_DIR
        / "03_stage3_kl.csv",
        index=False,
    )

    print_ranking(
        stage3_summary,
        "STAGE 3 — PRIOR KL",
    )

    # -------------------------------------------------------------------------
    # Combined tuning candidates
    # -------------------------------------------------------------------------

    all_tune_df = pd.DataFrame(
        all_tune_rows
    )

    all_tune_df.to_csv(
        OUTPUT_DIR
        / "04_all_tune_per_seed.csv",
        index=False,
    )

    all_tune_summary = aggregate_config_rows(
        all_tune_df,
        BASE_CFG,
    )

    all_tune_summary.to_csv(
        OUTPUT_DIR
        / "05_all_tune_summary.csv",
        index=False,
    )

    print_ranking(
        all_tune_summary,
        "COMBINED TUNE RANKING",
        n=15,
    )

    # -------------------------------------------------------------------------
    # B) INDEPENDENT VERIFICATION
    # -------------------------------------------------------------------------

    verify_candidates = (
        select_top(
            all_tune_summary,
            max(
                VERIFY_TOP_K * 2,
                VERIFY_TOP_K
            ),
        )
    )

    verify_candidates = (
        deduplicate_candidate_rows(
            verify_candidates
        )
        .head(VERIFY_TOP_K)
    )

    verify_cache = build_seed_cache(
        VERIFY_SEEDS,
        BASE_CFG,
        "VERIFY",
    )

    verify_rows = []

    for i, (_, candidate) in enumerate(
        verify_candidates.iterrows(),
        start=1,
    ):
        cfg = config_from_row(
            candidate
        )

        cfg_id = f"VERIFY_{i:02d}"

        rows = evaluate_config_on_cache(
            cfg_id,
            "Verify",
            cfg,
            verify_cache,
        )

        verify_rows.extend(rows)

    verify_df = pd.DataFrame(
        verify_rows
    )

    verify_df.to_csv(
        OUTPUT_DIR
        / "06_verify_per_seed.csv",
        index=False,
    )

    verify_summary = aggregate_config_rows(
        verify_df,
        BASE_CFG,
    )

    verify_summary.to_csv(
        OUTPUT_DIR
        / "07_verify_summary.csv",
        index=False,
    )

    print_ranking(
        verify_summary,
        "INDEPENDENT VERIFICATION RANKING",
        n=VERIFY_TOP_K,
    )

    winner_row = select_top(
        verify_summary,
        1,
    ).iloc[0]

    winner_cfg = config_from_row(
        winner_row
    )

    # -------------------------------------------------------------------------
    # Boundary check
    # -------------------------------------------------------------------------

    warnings = boundary_warnings(
        winner_row
    )

    print(
        "\n"
        + "=" * 160
    )
    print(
        "FROZEN WINNER AFTER VERIFICATION"
    )
    print(
        "=" * 160
    )

    print(
        json.dumps(
            {
                "lambda_fin":
                    winner_cfg.lambda_fin,

                "hybrid_delta":
                    winner_cfg.hybrid_delta,

                "prior_kl_weight":
                    winner_cfg.prior_kl_weight,

                "min_allocation":
                    winner_cfg.min_allocation,

                "verification_crps_delta_pct":
                    float(
                        winner_row[
                            "CRPS Delta %"
                        ]
                    ),

                "verification_tail_brier_delta":
                    float(
                        winner_row[
                            "Tail Brier Delta"
                        ]
                    ),

                "verification_stress_crps_delta":
                    float(
                        winner_row[
                            "Stress CRPS Delta"
                        ]
                    ),

                "verification_stress_tail_brier_delta":
                    float(
                        winner_row[
                            "Stress Tail Brier Delta"
                        ]
                    ),
            },
            indent=2,
        )
    )

    if warnings:
        print(
            "\nBOUNDARY WARNING:"
        )
        for w in warnings:
            print(f"  - {w}")

        print(
            "The holdout will still run, "
            "but for a publication-grade search you should "
            "consider expanding that boundary BEFORE treating "
            "the winner as permanently fixed."
        )
    else:
        print(
            "\nNo search-boundary warning."
        )

    # Save winner before holdout.
    winner_payload = {
        **asdict(winner_cfg),
        "run_profile": RUN_PROFILE,
        "tune_seeds": TUNE_SEEDS,
        "verify_seeds": VERIFY_SEEDS,
        "holdout_seeds": HOLDOUT_SEEDS,
        "boundary_warnings": warnings,
    }

    with open(
        OUTPUT_DIR
        / "08_frozen_winner_config.json",
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            winner_payload,
            f,
            indent=2,
        )

    # -------------------------------------------------------------------------
    # C) FINAL HOLDOUT
    # -------------------------------------------------------------------------

    holdout_cache = build_seed_cache(
        HOLDOUT_SEEDS,
        winner_cfg,
        "FINAL HOLDOUT",
    )

    final_df, allocation_df = run_final_holdout(
        winner_cfg,
        holdout_cache,
    )

    final_df.to_csv(
        OUTPUT_DIR
        / "09_final_holdout_per_seed.csv",
        index=False,
    )

    allocation_df.to_csv(
        OUTPUT_DIR
        / "10_final_holdout_allocations.csv",
        index=False,
    )

    final_summary = summarize_final_metrics(
        final_df
    )

    final_summary.to_csv(
        OUTPUT_DIR
        / "11_final_holdout_summary_mean_std.csv"
    )

    paired_report = make_final_paired_report(
        final_df
    )

    paired_report.to_csv(
        OUTPUT_DIR
        / "12_final_paired_bootstrap_report.csv",
        index=False,
    )

    if not allocation_df.empty:
        allocation_summary = (
            allocation_df
            .groupby(
                ["Method", "Risk Bin"]
            )[
                ["pi_D", "pi_N", "pi_T"]
            ]
            .agg(["mean", "std"])
        )

        allocation_summary.columns = [
            f"{a}_{b}"
            for a, b
            in allocation_summary.columns
        ]

        allocation_summary = (
            allocation_summary
            .reset_index()
        )

        allocation_summary.to_csv(
            OUTPUT_DIR
            / "13_final_allocation_by_risk.csv",
            index=False,
        )

    # -------------------------------------------------------------------------
    # D) FINAL DISPLAY
    # -------------------------------------------------------------------------

    print(
        "\n"
        + "=" * 160
    )
    print(
        "FINAL HOLDOUT RESULTS — MEAN ± STD"
    )
    print(
        "=" * 160
    )

    key_cols = [
        "CRPS",
        "Direction Brier",
        "Tail Brier",
        "Stress CRPS",
        "Stress Tail Brier",
        "Tail calibration gap",
        "Stress tail calibration gap",
    ]

    existing_key_cols = [
        c
        for c in key_cols
        if c in final_summary.columns
    ]

    print(
        final_summary[
            existing_key_cols
        ].to_string()
    )

    print(
        "\n"
        + "=" * 160
    )
    print(
        "FINAL PAIRED DELTAS VS CRPS"
    )
    print(
        "Negative delta = improvement. "
        "Bootstrap CI entirely below zero is a strong stability signal."
    )
    print(
        "=" * 160
    )

    print(
        paired_report
        .round(6)
        .to_string(index=False)
    )

    # Winner-specific compact verdict
    winner_final = paired_report[
        paired_report["Method"]
        == "monotonic_hybrid"
    ]

    print(
        "\n"
        + "=" * 160
    )
    print(
        "MONOTONIC HYBRID — FINAL HOLDOUT VERDICT"
    )
    print(
        "=" * 160
    )

    print(
        winner_final
        .round(6)
        .to_string(index=False)
    )

    print(
        "\nSaved output directory:"
    )
    print(
        OUTPUT_DIR.resolve()
    )

    print(
        "\nIMPORTANT INTERPRETATION:"
    )
    print(
        "The final holdout is evaluation only. "
        "Do not retune hyperparameters after seeing these holdout results. "
        "If the result is weak, define a new model version and use a new "
        "untouched holdout set."
    )

    print(
        "\nFinished."
    )


if __name__ == "__main__":
    main()
