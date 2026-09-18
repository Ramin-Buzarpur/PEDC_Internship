"""
Sanity check for SC-twCRPS v0.2

A scoring rule is "proper" if, in expectation over the TRUE data-generating
distribution, the true parameters minimize the score. We test this directly:

  1. Generate synthetic data from a KNOWN Gaussian(mu*, sigma*) where sigma*
     depends on a known exogenous regime variable (so heteroscedasticity is
     real, not just noise).
  2. Directly optimize free parameters (mu_hat, sigma_hat) per sample via
     gradient descent on the loss (no neural net -- isolates the loss itself
     from any confound of network capacity/architecture).
  3. Check:
       a) Does plain Gaussian CRPS recover (mu*, sigma*)? (expected: yes,
          CRPS is a known proper scoring rule -- this is our control.)
       b) Does SC-twCRPS (with alpha > 0) still recover (mu*, sigma*), or
          does it introduce systematic bias (e.g. inflating sigma in
          high-regime samples beyond the truth)? This is the real test.
       c) Does increasing alpha measurably improve tail coverage (VaR95/99)
          on a *deliberately misspecified* starting point, without wrecking
          CRPS on the correctly-specified case?
"""

import torch
from SC_twCRPS import gaussian_crps, sc_twcrps, SC_twCRPS_loss

torch.manual_seed(0)


def make_synthetic(n=2000):
    # exogenous regime: 0 = calm, 1 = crisis (known in advance, lagged)
    regime = torch.randint(0, 2, (n,)).float()
    true_mu = torch.zeros(n)
    true_sigma = torch.where(regime == 1, torch.tensor(2.0), torch.tensor(0.5))
    y = true_mu + true_sigma * torch.randn(n)
    return y, true_mu, true_sigma, regime


def fit(y, exog_center, exog_scale, exog_state, grid, alpha, lam,
        n_steps=800, lr=0.05, regime=None):
    """
    IMPORTANT: parameters are shared PER REGIME (2 values for mu, 2 for
    sigma), not one free (mu,sigma) per sample. A per-sample free parameter
    fit to a single observation is degenerate for CRPS (it collapses sigma
    -> 0 and mu -> y, which trivially "wins" without learning anything --
    confirmed by direct test). Sharing parameters within each regime is what
    makes recovery of the true regime-conditional sigma an identifiable,
    meaningful test.
    """
    n = y.shape[0]
    regime = regime.long()
    n_regimes = int(regime.max().item()) + 1

    mu_param = torch.zeros(n_regimes, requires_grad=True)
    log_sigma_param = torch.zeros(n_regimes, requires_grad=True)

    opt = torch.optim.Adam([mu_param, log_sigma_param], lr=lr)

    for step in range(n_steps):
        opt.zero_grad()
        sigma_param = torch.nn.functional.softplus(log_sigma_param) + 1e-3
        mu_hat = mu_param[regime]
        sigma_hat = sigma_param[regime]

        if alpha == 0.0:
            loss = gaussian_crps(mu_hat, sigma_hat, y)
        else:
            loss = SC_twCRPS_loss(mu_hat, sigma_hat, y, exog_center,
                                   exog_scale, exog_state, grid,
                                   lam=lam, alpha=alpha)
        loss.backward()
        opt.step()

    sigma_param = torch.nn.functional.softplus(log_sigma_param) + 1e-3
    mu_hat = mu_param[regime].detach()
    sigma_hat = sigma_param[regime].detach()
    return mu_hat, sigma_hat


def var95_coverage(y, mu, sigma):
    # empirical coverage of the model's own 95% interval
    from math import erf, sqrt
    z95 = 1.6448536269514722  # one-sided 95% normal quantile
    upper = mu + z95 * sigma
    lower = mu - z95 * sigma
    inside = ((y >= lower) & (y <= upper)).float().mean().item()
    return inside


def main():
    y, true_mu, true_sigma, regime = make_synthetic()
    grid = torch.linspace(-4, 4, 200)

    # Exogenous center/scale: a NOISY, imperfect proxy for the true regime
    # scale (e.g. what a rolling realized-vol estimator would give you in
    # practice) -- deliberately NOT equal to true_sigma and NOT derived from
    # the model being fit, only from the (known) regime label plus noise.
    exog_center = torch.zeros_like(y)
    exog_scale = (true_sigma + 0.1 * torch.randn_like(true_sigma)).clamp(min=0.1)
    exog_state = regime  # exogenous regime signal driving the tail boost

    print("=" * 70)
    print("TEST A: control -- does plain Gaussian CRPS recover ground truth?")
    print("=" * 70)
    mu_hat, sigma_hat = fit(y, exog_center, exog_scale, exog_state, grid,
                             alpha=0.0, lam=0.0, regime=regime)
    mu_err = (mu_hat - true_mu).abs().mean().item()
    sigma_err = (sigma_hat - true_sigma).abs().mean().item()
    print(f"  mean |mu_hat - mu*|       = {mu_err:.4f}  (want: near 0)")
    print(f"  mean |sigma_hat - sigma*| = {sigma_err:.4f}  (want: near 0)")
    print(f"  95% interval coverage     = {var95_coverage(y, mu_hat, sigma_hat):.3f} (want: ~0.95)")

    print()
    print("=" * 70)
    print("TEST B: does SC-twCRPS (alpha>0) still recover ground truth")
    print("        when the model IS correctly specified? (propriety check)")
    print("=" * 70)
    for alpha in [0.1, 0.5, 1.0, 2.0]:
        mu_hat, sigma_hat = fit(y, exog_center, exog_scale, exog_state, grid,
                                 alpha=alpha, lam=1.0, regime=regime)
        mu_err = (mu_hat - true_mu).abs().mean().item()
        sigma_err = (sigma_hat - true_sigma).abs().mean().item()
        # bias broken down by regime -- this is where SAFPS-style distortion
        # would show up (over-inflating sigma specifically in crisis regime)
        calm_bias = (sigma_hat[regime == 0] - true_sigma[regime == 0]).mean().item()
        crisis_bias = (sigma_hat[regime == 1] - true_sigma[regime == 1]).mean().item()
        cov = var95_coverage(y, mu_hat, sigma_hat)
        print(f"  alpha={alpha:>4}: |mu_err|={mu_err:.4f}  |sigma_err|={sigma_err:.4f}  "
              f"calm_sigma_bias={calm_bias:+.4f}  crisis_sigma_bias={crisis_bias:+.4f}  "
              f"cov95={cov:.3f}")

    print()
    print("=" * 70)
    print("TEST C: tail-focused benefit under MISSPECIFICATION")
    print("        (start from an under-confident-in-crisis initial fit and")
    print("         see whether SC-twCRPS pulls crisis-sigma toward truth")
    print("         faster / more than plain CRPS within a fixed step budget)")
    print("=" * 70)
    for alpha in [0.0, 1.0]:
        mu_hat, sigma_hat = fit(y, exog_center, exog_scale, exog_state, grid,
                                 alpha=alpha, lam=1.0, n_steps=150, regime=regime)
        crisis_err = (sigma_hat[regime == 1] - true_sigma[regime == 1]).abs().mean().item()
        calm_err = (sigma_hat[regime == 0] - true_sigma[regime == 0]).abs().mean().item()
        print(f"  alpha={alpha}: crisis |sigma_err| (150 steps) = {crisis_err:.4f}   "
              f"calm |sigma_err| = {calm_err:.4f}")

    print()
    print("Interpretation guide:")
    print(" - Test A should show both errors close to 0 (control passes).")
    print(" - Test B: if sigma_err grows a lot with alpha, or crisis_bias is")
    print("   consistently positive/large while calm_bias stays small, the")
    print("   loss is NOT proper in practice -- it's systematically distorting")
    print("   sigma estimates in the tail regime, same failure mode as SAFPS.")
    print(" - Test C: if alpha=1.0 converges faster on crisis sigma without")
    print("   hurting calm sigma, that's the actual selling point of this loss.")


if __name__ == "__main__":
    main()