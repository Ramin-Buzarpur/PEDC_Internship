
"""
SAFPS v0.1.3
Sensitivity & Tuning Framework for Monotonic Hybrid Controller

Purpose
-------
Tune the v0.1.x core BEFORE moving to v0.2.

This script:
1) Builds the same synthetic financial environment.
2) Trains a CRPS baseline once per development seed.
3) Tunes Monotonic Hybrid SAFPS in three stages:
      Stage 1: lambda_fin x hybrid_delta
      Stage 2: prior_kl_weight
      Stage 3: min_allocation
4) Applies research constraints:
      - zero monotonicity violations
      - CRPS degradation <= allowed threshold
5) Ranks configurations primarily by Stress Tail Brier.
6) Saves all results to CSV.

IMPORTANT
---------
The development seeds used here are for tuning.
Do NOT treat them as final test seeds.

After choosing a final configuration, test it ONCE
on completely new seeds (e.g. 1000-1029).

Requirements
------------
pip install torch numpy pandas

Run
---
python SAFPS_v013.py
"""

import math
import random
from copy import deepcopy
from dataclasses import dataclass, replace

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================
# USER SETTINGS
# ============================================================

# "quick" = faster exploratory tuning
# "full"  = more serious tuning
RUN_MODE = "quick"


# ============================================================
# Base configuration
# ============================================================

@dataclass
class Config:
    # Data
    n_samples: int = 7000
    train_size: int = 4500
    val_size: int = 1200

    # Training
    epochs: int = 45
    batch_size: int = 256
    lr: float = 3e-3

    # Weighted-CRPS integration
    z_min: float = -8.0
    z_max: float = 8.0
    n_grid: int = 129

    # Tuned parameters
    lambda_fin: float = 2.0
    hybrid_delta: float = 0.35
    prior_kl_weight: float = 0.02
    min_allocation: float = 0.05

    # Financial regions
    direction_width: float = 0.45
    downside_temp: float = 0.55
    tail_threshold: float = 2.2
    tail_temp: float = 0.35

    # Diagnostics
    stress_threshold: float = 0.70
    monotonic_tol: float = 1e-7

    # Selection constraint:
    # adaptive CRPS may not be worse than baseline
    # by more than this percentage.
    max_crps_degradation_pct: float = 0.10


BASE_CFG = Config()

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# ============================================================
# Tuning space
# ============================================================

if RUN_MODE == "quick":
    DEV_SEEDS = [101, 102, 103]

    STAGE1_LAMBDAS = [1.0, 1.5, 2.0, 2.5]
    STAGE1_DELTAS = [0.15, 0.35, 0.50]

    STAGE2_KLS = [0.00, 0.02, 0.05]
    STAGE3_MIN_ALLOC = [0.02, 0.05, 0.08]

    TOP_K_STAGE1 = 3
    TOP_K_STAGE2 = 2

else:
    DEV_SEEDS = list(range(101, 111))

    STAGE1_LAMBDAS = [0.5, 1.0, 1.5, 2.0, 3.0]
    STAGE1_DELTAS = [0.10, 0.20, 0.35, 0.50]

    STAGE2_KLS = [0.00, 0.01, 0.02, 0.05]
    STAGE3_MIN_ALLOC = [0.02, 0.05, 0.08, 0.10]

    TOP_K_STAGE1 = 5
    TOP_K_STAGE2 = 3


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
        [
            signal,
            risk_state,
            dir_uncertainty,
        ],
        axis=1,
    ).astype(np.float32)

    return X, y.astype(np.float32)


# ============================================================
# Forecast model
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


# ============================================================
# Allocation helper
# ============================================================

def bounded_simplex(logits, min_allocation):
    k = logits.shape[1]

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
# Financial prior
# ============================================================

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


# ============================================================
# Monotonic block
# ============================================================

class Monotonic1D(nn.Module):
    def __init__(self, hidden=6):
        super().__init__()

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
        w1 = F.softplus(
            self.raw_w1
        )

        w2 = F.softplus(
            self.raw_w2
        )

        hidden = F.softplus(
            x[:, None]
            * w1[None, :]
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
    def __init__(
        self,
        delta,
        min_allocation,
    ):
        super().__init__()

        self.delta = delta
        self.min_allocation = min_allocation

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

        return bounded_simplex(
            base_logits + correction,
            self.min_allocation,
        )


# ============================================================
# Weighted CRPS
# ============================================================

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
# Data preparation
# ============================================================

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

    data = {
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

    return data


# ============================================================
# Metrics
# ============================================================

@torch.no_grad()
def evaluate_forecast(
    model,
    controller,
    X,
    y,
    cfg,
):
    model.eval()
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

    threshold = (
        -cfg.tail_threshold
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

    stress_mask = (
        X[:, 1]
        > cfg.stress_threshold
    )

    pi = controller(X)

    return {
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

        "Mean pi_D":
            pi[:, 0].mean().item(),

        "Mean pi_N":
            pi[:, 1].mean().item(),

        "Mean pi_T":
            pi[:, 2].mean().item(),
    }


@torch.no_grad()
def evaluate_crps_baseline(
    model,
    X,
    y,
    cfg,
):
    model.eval()

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

    return {
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


# ============================================================
# Monotonicity diagnostics
# ============================================================

@torch.no_grad()
def monotonicity_diagnostics(
    controller,
    cfg,
):
    controller.eval()

    state_grid = torch.linspace(
        0.0,
        1.0,
        101,
        device=DEVICE,
    )

    # Risk sweep
    x_risk = torch.zeros(
        (101, 3),
        device=DEVICE,
    )

    x_risk[:, 1] = state_grid
    x_risk[:, 2] = 0.5

    pi_risk = controller(
        x_risk
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

    # Direction sweep
    x_dir = torch.zeros(
        (101, 3),
        device=DEVICE,
    )

    x_dir[:, 1] = 0.5
    x_dir[:, 2] = state_grid

    pi_dir = controller(
        x_dir
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


# ============================================================
# Train baseline
# ============================================================

def train_crps_baseline(
    seed,
    data,
    base_model_state,
    cfg,
):
    model = GaussianForecaster().to(
        DEVICE
    )

    model.load_state_dict(
        deepcopy(
            base_model_state
        )
    )

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=cfg.lr,
    )

    best_state = None
    best_val = float("inf")

    n = len(
        data["X_train"]
    )

    for epoch in range(
        cfg.epochs
    ):
        model.train()

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
            cfg.batch_size,
        ):
            idx = order[
                start:
                start + cfg.batch_size
            ].to(DEVICE)

            xb = data[
                "X_train"
            ][idx]

            yb = data[
                "y_train"
            ][idx]

            mu, sigma = model(
                xb
            )

            loss = gaussian_crps(
                mu,
                sigma,
                yb,
            ).mean()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        model.eval()

        with torch.no_grad():
            mu_v, sigma_v = model(
                data["X_val"]
            )

            val = gaussian_crps(
                mu_v,
                sigma_v,
                data["y_val"],
            ).mean().item()

        if val < best_val:
            best_val = val

            best_state = deepcopy(
                model.state_dict()
            )

    model.load_state_dict(
        best_state
    )

    return model


# ============================================================
# Train one monotonic-hybrid configuration
# ============================================================

def train_monotonic_config(
    seed,
    data,
    base_model_state,
    cfg,
):
    model = GaussianForecaster().to(
        DEVICE
    )

    model.load_state_dict(
        deepcopy(
            base_model_state
        )
    )

    torch.manual_seed(
        seed * 1000 + 777
    )

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(
            seed * 1000 + 777
        )

    controller = MonotonicHybridController(
        delta=cfg.hybrid_delta,
        min_allocation=cfg.min_allocation,
    ).to(DEVICE)

    params = (
        list(model.parameters())
        + list(controller.parameters())
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

    best_state = None
    best_val = float("inf")

    n = len(
        data["X_train"]
    )

    for epoch in range(
        cfg.epochs
    ):
        model.train()
        controller.train()

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
            cfg.batch_size,
        ):
            idx = order[
                start:
                start + cfg.batch_size
            ].to(DEVICE)

            xb = data[
                "X_train"
            ][idx]

            yb = data[
                "y_train"
            ][idx]

            mu, sigma = model(
                xb
            )

            pi = controller(
                xb
            )

            score = adaptive_weighted_crps(
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

            prior = financial_prior(
                xb,
                cfg,
            )

            kl = (
                pi
                * (
                    torch.log(
                        pi + 1e-8
                    )
                    - torch.log(
                        prior + 1e-8
                    )
                )
            ).sum(
                dim=1
            ).mean()

            loss = (
                score
                + cfg.prior_kl_weight
                * kl
            )

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # validation objective
        model.eval()
        controller.eval()

        with torch.no_grad():
            mu_v, sigma_v = model(
                data["X_val"]
            )

            pi_v = controller(
                data["X_val"]
            )

            score_v = adaptive_weighted_crps(
                mu_v,
                sigma_v,
                data["y_val"],
                pi_v,
                grid,
                qd,
                qn,
                qt,
                cfg.lambda_fin,
            ).mean()

            prior_v = financial_prior(
                data["X_val"],
                cfg,
            )

            kl_v = (
                pi_v
                * (
                    torch.log(
                        pi_v + 1e-8
                    )
                    - torch.log(
                        prior_v + 1e-8
                    )
                )
            ).sum(
                dim=1
            ).mean()

            val = (
                score_v
                + cfg.prior_kl_weight
                * kl_v
            ).item()

        if val < best_val:
            best_val = val

            best_state = {
                "model":
                    deepcopy(
                        model.state_dict()
                    ),

                "controller":
                    deepcopy(
                        controller.state_dict()
                    ),
            }

    model.load_state_dict(
        best_state["model"]
    )

    controller.load_state_dict(
        best_state["controller"]
    )

    return model, controller


# ============================================================
# Cache development data and baselines
# ============================================================

def build_dev_cache():
    cache = {}

    print(
        "\nPreparing development seeds "
        "and CRPS baselines..."
    )

    for seed in DEV_SEEDS:
        print(
            f"  Baseline seed {seed}"
        )

        set_global_seed(
            seed
        )

        data = prepare_seed_data(
            seed,
            BASE_CFG,
        )

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

        baseline_model = train_crps_baseline(
            seed,
            data,
            base_model_state,
            BASE_CFG,
        )

        baseline_metrics = evaluate_crps_baseline(
            baseline_model,
            data["X_test"],
            data["y_test"],
            BASE_CFG,
        )

        cache[seed] = {
            "data":
                data,

            "base_model_state":
                base_model_state,

            "baseline_metrics":
                baseline_metrics,
        }

    return cache


# ============================================================
# Evaluate one hyperparameter configuration
# ============================================================

def evaluate_configuration(
    config_id,
    stage,
    cfg,
    cache,
):
    rows = []

    print(
        f"\n[{stage}] {config_id}"
        f" | lambda={cfg.lambda_fin}"
        f" delta={cfg.hybrid_delta}"
        f" kl={cfg.prior_kl_weight}"
        f" min_alloc={cfg.min_allocation}"
    )

    for seed in DEV_SEEDS:
        item = cache[
            seed
        ]

        model, controller = train_monotonic_config(
            seed,
            item["data"],
            item["base_model_state"],
            cfg,
        )

        metrics = evaluate_forecast(
            model,
            controller,
            item["data"]["X_test"],
            item["data"]["y_test"],
            cfg,
        )

        mono = monotonicity_diagnostics(
            controller,
            cfg,
        )

        baseline = item[
            "baseline_metrics"
        ]

        crps_delta = (
            metrics["CRPS"]
            - baseline["CRPS"]
        )

        crps_delta_pct = (
            100.0
            * crps_delta
            / baseline["CRPS"]
        )

        row = {
            "Stage":
                stage,

            "Config ID":
                config_id,

            "Seed":
                seed,

            "lambda_fin":
                cfg.lambda_fin,

            "hybrid_delta":
                cfg.hybrid_delta,

            "prior_kl_weight":
                cfg.prior_kl_weight,

            "min_allocation":
                cfg.min_allocation,

            **metrics,
            **mono,

            "Baseline CRPS":
                baseline["CRPS"],

            "Baseline Direction Brier":
                baseline["Direction Brier"],

            "Baseline Tail Brier":
                baseline["Tail Brier"],

            "Baseline Stress CRPS":
                baseline["Stress CRPS"],

            "Baseline Stress Tail Brier":
                baseline[
                    "Stress Tail Brier"
                ],

            "CRPS Delta":
                crps_delta,

            "CRPS Delta %":
                crps_delta_pct,

            "Direction Brier Delta":
                (
                    metrics["Direction Brier"]
                    - baseline["Direction Brier"]
                ),

            "Tail Brier Delta":
                (
                    metrics["Tail Brier"]
                    - baseline["Tail Brier"]
                ),

            "Stress CRPS Delta":
                (
                    metrics["Stress CRPS"]
                    - baseline["Stress CRPS"]
                ),

            "Stress Tail Brier Delta":
                (
                    metrics["Stress Tail Brier"]
                    - baseline[
                        "Stress Tail Brier"
                    ]
                ),
        }

        rows.append(
            row
        )

    return rows


# ============================================================
# Aggregate configurations
# ============================================================

def aggregate_configs(
    per_seed_df,
    cfg_template,
):
    metrics = [
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
        "Risk monotonic violations",
        "Dir monotonic violations",
        "Delta pi_T risk 0->1",
        "Delta pi_D dir 0->1",
        "CRPS Delta",
        "CRPS Delta %",
        "Direction Brier Delta",
        "Tail Brier Delta",
        "Stress CRPS Delta",
        "Stress Tail Brier Delta",
    ]

    group_cols = [
        "Stage",
        "Config ID",
        "lambda_fin",
        "hybrid_delta",
        "prior_kl_weight",
        "min_allocation",
    ]

    mean_df = (
        per_seed_df
        .groupby(
            group_cols
        )[metrics]
        .mean()
        .reset_index()
    )

    std_df = (
        per_seed_df
        .groupby(
            group_cols
        )[metrics]
        .std(
            ddof=1
        )
        .reset_index()
    )

    std_df = std_df.rename(
        columns={
            col: f"{col} STD"
            for col in metrics
        }
    )

    summary = mean_df.merge(
        std_df,
        on=group_cols,
        how="left",
    )

    # Research constraints
    summary[
        "Monotonic OK"
    ] = (
        (
            summary[
                "Risk monotonic violations"
            ]
            == 0
        )
        &
        (
            summary[
                "Dir monotonic violations"
            ]
            == 0
        )
    )

    summary[
        "CRPS Constraint OK"
    ] = (
        summary[
            "CRPS Delta %"
        ]
        <= cfg_template.max_crps_degradation_pct
    )

    summary[
        "Eligible"
    ] = (
        summary[
            "Monotonic OK"
        ]
        &
        summary[
            "CRPS Constraint OK"
        ]
    )

    # Rank:
    # 1) eligible first
    # 2) lower Stress Tail Brier
    # 3) lower Tail Brier
    # 4) lower Stress CRPS
    # 5) lower CRPS
    summary = summary.sort_values(
        by=[
            "Eligible",
            "Stress Tail Brier",
            "Tail Brier",
            "Stress CRPS",
            "CRPS",
        ],
        ascending=[
            False,
            True,
            True,
            True,
            True,
        ],
    ).reset_index(
        drop=True
    )

    summary[
        "Rank"
    ] = np.arange(
        1,
        len(summary) + 1
    )

    return summary


# ============================================================
# Select top configs
# ============================================================

def select_top_configs(
    summary,
    top_k,
):
    eligible = summary[
        summary["Eligible"]
    ].copy()

    if len(eligible) == 0:
        print(
            "\nWARNING: no eligible config. "
            "Falling back to overall ranking."
        )

        eligible = summary.copy()

    return eligible.head(
        top_k
    )


# ============================================================
# Build a Config from row
# ============================================================

def cfg_from_row(row):
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


# ============================================================
# Print top table
# ============================================================

def print_top_table(
    summary,
    title,
    n=10,
):
    print(
        "\n"
        + "=" * 150
    )

    print(title)

    print(
        "=" * 150
    )

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
        "Stress Tail Brier",
        "Risk monotonic violations",
        "Dir monotonic violations",
    ]

    print(
        summary[
            cols
        ]
        .head(n)
        .round(6)
        .to_string(
            index=False
        )
    )


# ============================================================
# Main tuning pipeline
# ============================================================

def main():
    pd.set_option(
        "display.max_columns",
        None
    )

    pd.set_option(
        "display.width",
        280
    )

    print("=" * 150)
    print("SAFPS v0.1.3 — SENSITIVITY & TUNING")
    print("=" * 150)

    print(
        f"Run mode: {RUN_MODE}"
    )

    print(
        f"Device: {DEVICE}"
    )

    print(
        f"Development seeds: {DEV_SEEDS}"
    )

    print(
        "CRPS degradation limit: "
        f"{BASE_CFG.max_crps_degradation_pct:.3f}%"
    )

    print("=" * 150)

    cache = build_dev_cache()

    all_rows = []

    # ========================================================
    # STAGE 1
    # lambda_fin x hybrid_delta
    # ========================================================

    stage1_rows = []

    config_counter = 1

    for lam in STAGE1_LAMBDAS:
        for delta in STAGE1_DELTAS:
            cfg = replace(
                BASE_CFG,
                lambda_fin=lam,
                hybrid_delta=delta,
            )

            config_id = (
                f"S1_{config_counter:02d}"
            )

            rows = evaluate_configuration(
                config_id,
                "Stage1",
                cfg,
                cache,
            )

            stage1_rows.extend(
                rows
            )

            all_rows.extend(
                rows
            )

            config_counter += 1

    stage1_df = pd.DataFrame(
        stage1_rows
    )

    stage1_summary = aggregate_configs(
        stage1_df,
        BASE_CFG,
    )

    print_top_table(
        stage1_summary,
        "STAGE 1 RESULTS — lambda_fin x hybrid_delta",
    )

    stage1_summary.to_csv(
        "safps_v013_stage1_summary.csv",
        index=False,
    )

    top_stage1 = select_top_configs(
        stage1_summary,
        TOP_K_STAGE1,
    )

    # ========================================================
    # STAGE 2
    # Tune prior KL
    # ========================================================

    stage2_rows = []

    config_counter = 1

    for _, parent in top_stage1.iterrows():
        for kl in STAGE2_KLS:
            cfg = replace(
                BASE_CFG,
                lambda_fin=float(
                    parent["lambda_fin"]
                ),
                hybrid_delta=float(
                    parent["hybrid_delta"]
                ),
                prior_kl_weight=kl,
                min_allocation=float(
                    parent["min_allocation"]
                ),
            )

            config_id = (
                f"S2_{config_counter:02d}"
            )

            rows = evaluate_configuration(
                config_id,
                "Stage2",
                cfg,
                cache,
            )

            stage2_rows.extend(
                rows
            )

            all_rows.extend(
                rows
            )

            config_counter += 1

    stage2_df = pd.DataFrame(
        stage2_rows
    )

    stage2_summary = aggregate_configs(
        stage2_df,
        BASE_CFG,
    )

    print_top_table(
        stage2_summary,
        "STAGE 2 RESULTS — prior_kl_weight",
    )

    stage2_summary.to_csv(
        "safps_v013_stage2_summary.csv",
        index=False,
    )

    top_stage2 = select_top_configs(
        stage2_summary,
        TOP_K_STAGE2,
    )

    # ========================================================
    # STAGE 3
    # Tune min allocation
    # ========================================================

    stage3_rows = []

    config_counter = 1

    for _, parent in top_stage2.iterrows():
        for min_alloc in STAGE3_MIN_ALLOC:
            cfg = replace(
                BASE_CFG,
                lambda_fin=float(
                    parent["lambda_fin"]
                ),
                hybrid_delta=float(
                    parent["hybrid_delta"]
                ),
                prior_kl_weight=float(
                    parent["prior_kl_weight"]
                ),
                min_allocation=min_alloc,
            )

            config_id = (
                f"S3_{config_counter:02d}"
            )

            rows = evaluate_configuration(
                config_id,
                "Stage3",
                cfg,
                cache,
            )

            stage3_rows.extend(
                rows
            )

            all_rows.extend(
                rows
            )

            config_counter += 1

    stage3_df = pd.DataFrame(
        stage3_rows
    )

    stage3_summary = aggregate_configs(
        stage3_df,
        BASE_CFG,
    )

    print_top_table(
        stage3_summary,
        "STAGE 3 RESULTS — min_allocation",
    )

    stage3_summary.to_csv(
        "safps_v013_stage3_summary.csv",
        index=False,
    )

    # ========================================================
    # Final development ranking
    # ========================================================

    all_df = pd.DataFrame(
        all_rows
    )

    all_df.to_csv(
        "safps_v013_all_per_seed_results.csv",
        index=False,
    )

    all_summary = aggregate_configs(
        all_df,
        BASE_CFG,
    )

    all_summary.to_csv(
        "safps_v013_all_config_summary.csv",
        index=False,
    )

    print_top_table(
        all_summary,
        "FINAL DEVELOPMENT RANKING",
        n=15,
    )

    best = select_top_configs(
        all_summary,
        1,
    ).iloc[0]

    print(
        "\n"
        + "=" * 150
    )

    print(
        "BEST DEVELOPMENT CONFIGURATION"
    )

    print(
        "=" * 150
    )

    print(
        f"Config ID          : "
        f"{best['Config ID']}"
    )

    print(
        f"Stage              : "
        f"{best['Stage']}"
    )

    print(
        f"lambda_fin         : "
        f"{best['lambda_fin']}"
    )

    print(
        f"hybrid_delta       : "
        f"{best['hybrid_delta']}"
    )

    print(
        f"prior_kl_weight    : "
        f"{best['prior_kl_weight']}"
    )

    print(
        f"min_allocation     : "
        f"{best['min_allocation']}"
    )

    print(
        f"CRPS Delta %       : "
        f"{best['CRPS Delta %']:.6f}"
    )

    print(
        f"Tail Brier Delta   : "
        f"{best['Tail Brier Delta']:.6f}"
    )

    print(
        f"Stress CRPS Delta  : "
        f"{best['Stress CRPS Delta']:.6f}"
    )

    print(
        f"Stress Tail Delta  : "
        f"{best['Stress Tail Brier Delta']:.6f}"
    )

    print(
        f"Risk violations    : "
        f"{best['Risk monotonic violations']:.0f}"
    )

    print(
        f"Direction violations: "
        f"{best['Dir monotonic violations']:.0f}"
    )

    print(
        "\nIMPORTANT:"
    )

    print(
        "This is the best configuration on DEVELOPMENT seeds only."
    )

    print(
        "Do not claim final performance yet."
    )

    print(
        "Next step: freeze this configuration and evaluate "
        "once on completely unseen seeds."
    )

    print(
        "\nSaved files:"
    )

    print(
        "1) safps_v013_stage1_summary.csv"
    )

    print(
        "2) safps_v013_stage2_summary.csv"
    )

    print(
        "3) safps_v013_stage3_summary.csv"
    )

    print(
        "4) safps_v013_all_per_seed_results.csv"
    )

    print(
        "5) safps_v013_all_config_summary.csv"
    )

    print(
        "\nFinished."
    )


if __name__ == "__main__":
    main()
