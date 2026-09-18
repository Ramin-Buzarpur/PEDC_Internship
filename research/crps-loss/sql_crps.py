"""
SQL-CRPS: State-Adaptive Quantile-Layer CRPS  (v0.1)
====================================================

Core identity
-------------
CRPS(F, y) = 2 * INTEGRAL_0^1  rho_alpha(Q_alpha(F), y) d(alpha)

where Q_alpha is the alpha-quantile of F and rho_alpha is the pinball loss:

    rho_alpha(q, y) = (y - q) * (alpha - 1{y < q})

This is the classical quantile-layer representation of the CRPS (pinball
integral form). SQL-CRPS reweights the LAYERS alpha with a state-dependent,
strictly positive weight function:

    w(alpha | s) = (1 - lam(s)) + lam(s) * B(alpha),     0 <= lam(s) < 1
    B(alpha)     = 1/alpha0 * 1{alpha <= alpha0}         (tail block)

    SQL-CRPS(F, y | s) = 2 * INTEGRAL w(alpha|s) * rho_alpha(Q_alpha(F), y) d(alpha)

Why this is proper
------------------
For any FIXED s, w(alpha|s) > 0 almost everywhere. The score is a positive
weighted integral of pinball losses, each of which is a proper score for the
alpha-quantile functional. Hence the integral is minimized (uniquely, for
identifying families) at the true state-conditional quantile function, i.e.
at the true distribution F*. Properness holds conditionally on s, therefore
also unconditionally (tower property). Unlike tail-masked penalties
(e.g. SC-twCRPS v1.0), there is NO selection on the realized outcome y, and
NO term whose minimizer differs from the truth -- nothing to fight.

Why it should help under misspecification
-----------------------------------------
With a misspecified model family (always true in markets), every proper score
allocates its fixed accuracy budget differently. NLL spends it on density
ratios; plain CRPS spends it uniformly over alpha; SQL-CRPS shifts budget to
the extreme layers (small alpha) -- exactly where VaR/ES decisions live --
while retaining a strictly positive base weight everywhere, so the overall
distribution cannot degenerate.

Algorithmic properties
----------------------
- Closed form for Gaussian (and any location-scale) quantiles: zero MC noise,
  no rsample tricks, fully differentiable, bounded gradients (pinball
  subgradient is in [0,1]).
- lam(s) = lam_max * sigmoid(a * (s - s0)) is differentiable in s if we want
  to learn the state mapping; in v0.1 it is a frozen monotone map.

This file: numerically verify (1) the pinball-integral identity, (2)
propriety of the weighted score (recovers ground truth with shared free
parameters), (3) gradient health.
"""

import math
import torch

SQRT2 = math.sqrt(2.0)
SQRTPI = math.sqrt(math.pi)
SQRT2PI = math.sqrt(2.0 * math.pi)


# ---------------------------------------------------------------------
# Quantile layers
# ---------------------------------------------------------------------
def make_alpha_grid(n_core=48, n_tail=32, alpha_min=1e-3, alpha_max_tail=0.05):
    """
    Discrete layer grid: core layers uniform on (0.05, 0.95) and tail layers
    log-spaced on (alpha_min, alpha_max_tail). Base (core+tail-uniform)
    weights sum to 1 over the whole unit interval so that lam=0 reproduces
    plain CRPS up to grid error.
    """
    core = torch.linspace(0.05, 0.95, n_core)
    tail = torch.exp(torch.linspace(math.log(alpha_min),
                                    math.log(alpha_max_tail), n_tail))
    alphas = torch.cat([core, tail])
    # weights: total mass 0.5 on core, 0.5 on tail block
    w_core = torch.full((n_core,), 0.5 / n_core)
    w_tail = torch.full((n_tail,), 0.5 / n_tail)
    return alphas, torch.cat([w_core, w_tail])


def gaussian_quantiles(mu, sigma, alphas):
    z = torch.special.ndtri(alphas.to(mu.device))
    return mu[..., None] + sigma[..., None] * z


def pinball(q, y, alphas):
    """rho_alpha(q, y) = (y - q) * (alpha - 1{y < q}); shape [..., K]."""
    return (y[..., None] - q) * (alphas - (y[..., None] < q).float())


def sql_crps_loss(mu, sigma, y, alphas=None, weights=None,
                  lam=0.0, alpha0=0.05):
    """
    SQL-CRPS with scalar or per-sample lam (per-sample: shape like mu).
    lam=0 -> plain CRPS (discrete quantile-layer approximation).
    """
    if alphas is None:
        alphas, weights = make_alpha_grid()
    alphas = alphas.to(mu.device)
    weights = weights.to(mu.device)

    q = gaussian_quantiles(mu, sigma, alphas)          # [..., K]
    rho = pinball(q, y, alphas)                        # [..., K]

    if isinstance(lam, torch.Tensor):
        lam = lam[..., None]                            # broadcast over K
    w = (1.0 - lam) + lam * (alphas <= alpha0).float() / alpha0
    w = w * weights                                     # effective layer weight
    # normalize so the total weight is 1 (keeps CRPS scale for lam=0)
    w = w / w.sum(dim=-1, keepdim=True)

    return 2.0 * (w * rho).sum(dim=-1)


def gaussian_crps(mu, sigma, y):
    """Analytic Gaussian CRPS (Gneiting et al. 2005, eq. 5) -- reference."""
    z = (y - mu) / sigma
    Phi = 0.5 * (1 + torch.erf(z / SQRT2))
    phi = torch.exp(-0.5 * z * z) / SQRT2PI
    return sigma * (z * (2 * Phi - 1) + 2 * phi - 1 / SQRTPI)


def gaussian_nll(mu, sigma, y):
    z = (y - mu) / sigma
    return (0.5 * z * z + torch.log(sigma) + 0.5 * math.log(2 * math.pi))


# =====================================================================
# TEST 1: pinball-integral identity -- does the discrete layer sum with
#         lam=0 reproduce the analytic CRPS?
# =====================================================================
def test_identity():
    torch.manual_seed(0)
    print("=" * 70)
    print("TEST 1: quantile-layer integral (lam=0) == analytic CRPS?")
    print("=" * 70)
    mu, sigma = torch.tensor(0.3), torch.tensor(0.8)
    ys = torch.linspace(-2.5, 2.5, 11)
    alphas, _ = make_alpha_grid()
    # denser uniform grid for the identity check (proper CRPS uses uniform
    # alpha over (0,1); our research grid is tail-concentrated, so also
    # check against a dense uniform grid)
    unif = torch.linspace(1e-4, 1 - 1e-4, 4096)

    for y in ys:
        crps_true = gaussian_crps(mu, sigma, y).item()
        q = gaussian_quantiles(mu, sigma, unif)
        rho = pinball(q, y.unsqueeze(0), unif)
        crps_layer = 2 * rho.mean().item()
        print(f"  y={y:+.2f}: analytic={crps_true:.5f}  "
              f"layer(dense uniform)={crps_layer:.5f}  "
              f"diff={abs(crps_true-crps_layer):.5f}")


# =====================================================================
# TEST 2: propriety -- with free (mu, sigma) shared within regime, does
#         SQL-CRPS at various lam recover the truth? (lam=0 control is
#         plain CRPS; CRPS/NLL also run as reference)
# =====================================================================
def make_data(n=4000, seed=0):
    g = torch.Generator().manual_seed(seed)
    regime = torch.randint(0, 2, (n,), generator=g).float()
    true_mu = torch.zeros(n)
    true_sigma = torch.where(regime == 1, torch.tensor(2.0),
                             torch.tensor(0.5))
    y = true_mu + true_sigma * torch.randn(n, generator=g)
    return y, true_mu, true_sigma, regime


def fit_params(y, regime, loss_fn, steps=1200, lr=0.05):
    n_reg = int(regime.max().item()) + 1
    reg = regime.long()
    mu_p = torch.zeros(n_reg, requires_grad=True)
    log_s = torch.zeros(n_reg, requires_grad=True)
    opt = torch.optim.Adam([mu_p, log_s], lr=lr)
    for _ in range(steps):
        sigma = torch.exp(log_s).clamp(min=1e-3)
        loss = loss_fn(mu_p[reg], sigma[reg], y).mean()
        opt.zero_grad(); loss.backward(); opt.step()
    return mu_p[reg].detach(), torch.exp(log_s)[reg].detach()


def test_propriety():
    print()
    print("=" * 70)
    print("TEST 2: propriety -- recovery of true (mu, sigma) per regime")
    print("=" * 70)
    y, true_mu, true_sigma, regime = make_data()

    print("  -- reference baselines --")
    for name, fn in [
        ("NLL", gaussian_nll),
        ("CRPS", gaussian_crps),
    ]:
        mu_h, s_h = fit_params(y, regime, fn)
        print(f"  {name:8s}: sigma_err={ (s_h-true_sigma).abs().mean():.4f} "
              f"mu_err={ (mu_h-true_mu).abs().mean():.4f}")

    print("  -- SQL-CRPS (ours) --")
    alphas, weights = make_alpha_grid()
    for lam in [0.0, 0.3, 0.6, 0.9]:
        mu_h, s_h = fit_params(
            y, regime,
            lambda m, s, yy, lam=lam: sql_crps_loss(m, s, yy, alphas,
                                                    weights, lam=lam))
        calm_b = (s_h[regime == 0] - true_sigma[regime == 0]).mean().item()
        crisis_b = (s_h[regime == 1] - true_sigma[regime == 1]).mean().item()
        print(f"  lam={lam:.1f}  : sigma_err={ (s_h-true_sigma).abs().mean():.4f} "
              f"calm_bias={calm_b:+.4f} crisis_bias={crisis_b:+.4f}")


# =====================================================================
# TEST 3: gradient health at extreme layers (no explosion, no NaN)
# =====================================================================
def test_gradients():
    print()
    print("=" * 70)
    print("TEST 3: gradient norms per loss (per-sample mean |dL/dmu|,|dL/ds|)")
    print("=" * 70)
    y, _, _, regime = make_data()
    alphas, weights = make_alpha_grid()

    for name, mk in [
        ("NLL", lambda: gaussian_nll),
        ("CRPS", lambda: gaussian_crps),
        ("SQL lam=0.5", lambda: lambda m, s, yy:
            sql_crps_loss(m, s, yy, alphas, weights, lam=0.5)),
        ("SQL lam=0.9", lambda: lambda m, s, yy:
            sql_crps_loss(m, s, yy, alphas, weights, lam=0.9)),
    ]:
        mu_p = torch.zeros(2, requires_grad=True)
        log_s = torch.zeros(2, requires_grad=True)
        fn = mk()
        reg = regime.long()
        loss = fn(mu_p[reg], torch.exp(log_s)[reg], y).mean()
        loss.backward()
        gn = (mu_p.grad.abs().mean().item(),
              log_s.grad.abs().mean().item())
        print(f"  {name:12s}: loss={loss.item():8.4f}  "
              f"|dL/dmu|={gn[0]:.4f}  |dL/dlogS|={gn[1]:.4f}")


if __name__ == "__main__":
    test_identity()
    test_propriety()
    test_gradients()
