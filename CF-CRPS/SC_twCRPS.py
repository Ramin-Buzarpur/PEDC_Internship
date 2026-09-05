"""
SC-twCRPS v0.2
State-Conditional Threshold-Weighted CRPS (corrected)

Fixes applied vs v0.1:
  1. Thresholds are now PER-SAMPLE (mu_i + sigma_i * grid), so the tail of
     every sample's own distribution is actually covered, regardless of scale.
  2. Weight is a function of STANDARDIZED distance (u - mu)/sigma, not of the
     raw threshold value |u|. Fixes the "wrong center" bug.
  3. Weight is a SMOOTH TAIL GATE (sigmoid cutoff), not a global exponential
     that grows from the center. Near-mean errors are no longer over-penalized
     just because volatility is high (this was the exact failure mode that
     hurt SAFPS's CRPS).
  4. Weight growth is clamped -> no exp() blow-up on large realized_vol.
  5. Single unified weight (1 + boost) instead of computing CRPS twice.
  6. sigma is clamped everywhere it is used, not just in one function.
  7. Optional Student-t NLL term for a heavy-tailed base density, combined
     with the (Gaussian-approximated) tail-weighted CRPS as an auxiliary
     calibration penalty -- NOT as the sole loss.

Honest caveat: the tail-weighted CRPS term still uses a Gaussian CDF
approximation for tractability (a closed-form differentiable Student-t CDF
requires the incomplete beta function, which is not integrated here). This
is a simplification, not a hidden bug -- treat it as an approximate
calibration penalty layered on top of a Student-t density, not as an exact
Student-t CRPS. If you need the exact version, integrate the Student-t pdf
numerically (cumulative trapezoid) instead of using gaussian_cdf below.
"""

import torch


# ---------------------------------------------------------------------------
# Gaussian CRPS (unchanged, verified correct against Gneiting et al. 2005)
# ---------------------------------------------------------------------------

def gaussian_cdf(x, mu, sigma):
    sigma = torch.clamp(sigma, min=1e-6)
    z = (x - mu) / sigma
    return 0.5 * (1.0 + torch.erf(z / torch.sqrt(torch.tensor(2.0, device=x.device))))


def gaussian_crps(mu, sigma, y):
    sigma = torch.clamp(sigma, min=1e-6)
    z = (y - mu) / sigma

    pdf = torch.exp(-0.5 * z ** 2) / torch.sqrt(
        torch.tensor(2.0 * torch.pi, device=y.device)
    )
    cdf = gaussian_cdf(y, mu, sigma)

    crps = sigma * (
        z * (2 * cdf - 1)
        + 2 * pdf
        - 1 / torch.sqrt(torch.tensor(torch.pi, device=y.device))
    )
    return crps.mean()


# ---------------------------------------------------------------------------
# Student-t NLL (heavy-tailed base density)
# ---------------------------------------------------------------------------

def student_t_nll(y, mu, sigma, nu):
    sigma = torch.clamp(sigma, min=1e-6)
    nu = torch.clamp(nu, min=2.0 + 1e-3)

    z = (y - mu) / sigma
    log_const = (
        torch.lgamma((nu + 1) / 2)
        - torch.lgamma(nu / 2)
        - 0.5 * torch.log(nu * torch.tensor(torch.pi, device=y.device))
    )
    log_term = -(nu + 1) / 2 * torch.log1p(z ** 2 / nu)
    log_lik = log_const + log_term - torch.log(sigma)
    return -log_lik.mean()


# ---------------------------------------------------------------------------
# Corrected SC-twCRPS
# ---------------------------------------------------------------------------

def smooth_tail_weight(z_abs, exog_state, lam=1.0, cutoff=1.5, sharpness=5.0,
                        max_boost_log=4.0):
    """
    z_abs        : |u - mu| / sigma, standardized distance from center.
                   Same shape as the threshold grid, broadcastable to (n, K).
    exog_state   : per-sample exogenous state (e.g. lagged realized vol).
                   MUST be computed from information available before y_t
                   and MUST NOT depend on the model's own output for t.
    cutoff       : standardized distance (in sigmas) beyond which the tail
                   gate starts turning on. Below this, weight stays ~1.
    sharpness    : how sharply the gate turns on around `cutoff`.
    max_boost_log: clamps lam * exog_state so exp() cannot blow up.
    """
    gate = torch.sigmoid(sharpness * (z_abs - cutoff))          # ~0 near center, ~1 in tail
    boost = torch.exp(torch.clamp(lam * exog_state.unsqueeze(-1),
                                   max=max_boost_log)) - 1.0     # bounded growth
    return 1.0 + boost * gate


def per_sample_thresholds(exog_center, exog_scale, grid):
    """
    CRITICAL: thresholds must be built from EXOGENOUS center/scale (e.g. a
    rolling historical mean/vol estimate), NEVER from mu_hat/sigma_hat (the
    model's own current output). If thresholds depend on the forecast being
    scored, the model can trivially shrink sigma_hat to shrink the whole
    evaluation grid and cheat the score down without actually calibrating
    anything -- this was verified empirically (see sanity_check.py, Test B
    on the v2-with-model-scaled-thresholds variant: sigma collapsed to ~0
    and coverage collapsed to 0.5%). Using a fixed/exogenous scale removes
    the forecast's ability to move the goalposts it's being judged against.

    grid : 1D standardized grid, e.g. torch.linspace(-4, 4, 200)
    returns thresholds of shape (n, K) = exog_center_i + exog_scale_i * grid_k
    """
    exog_scale = torch.clamp(exog_scale, min=1e-6)
    return exog_center.unsqueeze(-1) + exog_scale.unsqueeze(-1) * grid.unsqueeze(0)


def sc_twcrps(mu, sigma, y, exog_center, exog_scale, exog_state, grid,
              lam=1.0, cutoff=1.5, sharpness=5.0):
    """
    Corrected, tail-only threshold-weighted CRPS.

    exog_center, exog_scale : precomputed from information available BEFORE
        the forecast is made (e.g. rolling mean / realized vol of the last
        N observations). Must NOT be a function of mu, sigma (the forecast
        currently being scored) or this term stops being proper -- see the
        docstring above and sanity_check.py for the empirical proof.
    exog_state : per-sample regime/vol signal driving how much extra weight
        the tail gets (also exogenous, same constraint as above).
    """
    sigma = torch.clamp(sigma, min=1e-6)

    t = per_sample_thresholds(exog_center, exog_scale, grid)     # (n, K), exogenous scale
    z_abs = grid.abs().unsqueeze(0)                              # (1, K), fixed standardized grid

    F = gaussian_cdf(t, mu.unsqueeze(-1), sigma.unsqueeze(-1))   # (n, K) -- forecast still scored at these fixed points
    indicator = (y.unsqueeze(-1) <= t).float()                   # (n, K)
    error = (F - indicator) ** 2

    weight = smooth_tail_weight(z_abs, exog_state, lam=lam,
                                 cutoff=cutoff, sharpness=sharpness)  # (n, K)

    integral = torch.trapz(error * weight, grid, dim=-1) * exog_scale
    return integral.mean()


def SC_twCRPS_loss(mu, sigma, y, exog_center, exog_scale, exog_state, grid,
                    nu=None, lam=1.0, alpha=1.0, cutoff=1.5, sharpness=5.0):
    """
    Final combined loss.

    If `nu` is provided: Student-t NLL (main density fit, heavy-tailed)
                          + alpha * SC-twCRPS (auxiliary tail-calibration penalty)
    Else: Gaussian CRPS + alpha * SC-twCRPS  (v0.1-style, but corrected)

    exog_center/exog_scale/exog_state must all be computed from information
    available before the forecast (e.g. rolling historical stats), never
    from mu/sigma themselves.
    """
    tail_term = sc_twcrps(mu, sigma, y, exog_center, exog_scale, exog_state,
                          grid, lam=lam, cutoff=cutoff, sharpness=sharpness)

    if nu is not None:
        base_term = student_t_nll(y, mu, sigma, nu)
    else:
        base_term = gaussian_crps(mu, sigma, y)

    return base_term + alpha * tail_term


if __name__ == "__main__":
    n = 64
    mu = torch.randn(n)
    sigma = torch.rand(n) * 0.5 + 0.2
    y = mu + sigma * torch.randn(n)

    # Exogenous state/scale: MUST be from past info only (here just simulated,
    # decoupled from mu/sigma which are the things being fit).
    exog_center = torch.zeros(n)
    exog_scale = torch.rand(n) * 0.5 + 0.2
    exog_state = torch.rand(n)

    grid = torch.linspace(-4, 4, 200)

    loss_gaussian_base = SC_twCRPS_loss(mu, sigma, y, exog_center, exog_scale,
                                         exog_state, grid, lam=1.0, alpha=0.5)
    print("SC-twCRPS v0.2 (Gaussian base):", loss_gaussian_base.item())

    nu = torch.rand(n) * 10 + 3
    loss_studentt_base = SC_twCRPS_loss(mu, sigma, y, exog_center, exog_scale,
                                         exog_state, grid, nu=nu,
                                         lam=1.0, alpha=0.5)
    print("SC-twCRPS v0.2 (Student-t base):", loss_studentt_base.item())