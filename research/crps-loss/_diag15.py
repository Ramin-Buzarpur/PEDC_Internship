"""
Diag 15: find what mu*, sigma* minimize the tied pinball loss
It's clearly degenerate. So the pure pinball-with-tied-quantiles is NOT a
valid proper scoring rule for the distribution.

What IS proper: the QUANTILE SCORE (also called "quantile-weighted CRPS"
or the "_pinsker_ loss" in some forms). Specifically, the integral of pinball
losses over ALL quantile levels is exactly the CRPS:

    CRPS(F, y) = 2 * ∫ rho_alpha(F^{-1}(alpha), y) dalpha

 evaluated at all alpha, which is what makes CRPS strictly proper.

So instead of a single alpha, use a SMALL SET of alphas (a "quantile
portfolio") and combine their pinball losses. But this is just the CRPS
approximated by quantiles, which isn't new.

What's a genuinely NEW, PROPER, and TAIL-SENSITIVE loss?

Answer: change the INTEGRATING MEASURE in the quantile-score representation
of the CRPS in a way that reweights the TAIL QUANTILES more, but keep the
weights only alpha-dependent (NOT y- or F-dependent), and RENORMALIZE so the
total mass is 1. This is the "Quantile-Weighted CRPS" (QWC) family from
Gneiting & Ranjan (2011) -- their threshold- and quantile-weighted CRPS are
known to NOT be proper in general.

Instead, we use a different principle: MIXTURE PROPERNESS. Define:

    TCM-CRPS_beta(F, y) = (1 - beta) * CRPS(F, y) + beta * CRPS(F_tail, y)

where F_tail is the CONDITIONAL distribution of F restricted to the tail
region (a normalized CDF over e.g. {X < q_alpha(F)}), and y is in the tail.
Hmm, this isn't well-defined for y outside the tail.

Actually, the right approach for PROPER tail sensitivity is the
"Conditional Tail Expectation" (CTE) / "Expected Shortfall" scoring rule of
Nolde & Ziegel (2017) or Fissler & Ziegel (2016). Their joint
(VaR, ES) scoring rule is a proper scoring rule for the pair (VaR_alpha, ES_alpha):

    S(q, e, y) = (1{y <= q} - alpha) * q - 0.5 * 1{y<=q} * q^2/(...)...

Let me just use the Fissler-Ziegel (2016) "FSF loss" (FZ0 class):
    S(q, e, y) = ( 1{y <= q} - alpha ) * q - e * ( q - y ... )

The FZ0 score is:
    S(q, e, y) = (1{y<=q} - alpha)*q - e*( q - 1{y<=q}*y ) ... hmm I'd better
    look this up exactly. The "0-homogeneous" FZ score is:

    S(q, e, y) = (1{y > q} - alpha) * (q - y)  ... no.

Let me just look at the simplest known-proper (VaR, ES) score, from
Fissler & Ziegel (2016), "higher order" elicitability. The FZ0 score:

    S(q, e, y; alpha) = ( 1{y <= q} - alpha ) * ( q - y ) - 1{y<=q} * e + ...

OK let me just be honest and DERIVE one myself. Actually, there's a well-known
regret: FZ scores are tricky. Let me look at a cleaner reference:
Taylor (2019) JBST uses:  S = (1{y<=q} - alpha)*q - 1{y<=q}*e + y*e (approximately)

Actually, let me just try the simplest possible approach: use the actual
Nolde-Ziegel / FZ "FZ0" loss:

    S_FZ0(q, e, y) = (1{y <= q} - alpha) * (q - y) - (e - q) * 1{y <= q} + e * log(...)

I don't remember. Let me just DERIVE it by requiring:
- It's minimized (in expectation) at the true (q*, e*) = (VaR, ES).

Actually, a MUCH cleaner idea: use the fact that ES_alpha is an elicitable
-functional jointly with VaR_alpha under the FZ score, but there's an even
simpler approach: Elicit the ENTIRE TAIL DISTRIBUTION with CRPS on the tail.

**NEW IDEA -- "Tail-CDF CRPS":**
Define the tail-restricted predictive distribution F_T, which is F conditioned
on {X <= q_alpha(F)} (or {X >= q_{1-alpha}}), normalized to be a proper CDF on
that region. For an observation y that falls in the tail (y <= q_alpha(F)),
score the conditional forecast with CRPS:

    CRPS_tail(F, y) = CRPS(F_T, y) restricted to the tail region.

If y is in the tail, this is a proper scoring rule for the tail shape, i.e.
for the conditional distribution of X given X <= VaR. This IS proper, because
CRPS is proper and conditioning on an F-determined event is like a
sub-distribution scoring rule.

Actually simpler and cleaner still: for y in the tail, F_T(y) = F(y)/alpha
is the PIT of the tail CDF. We want to check: does E_y~tail[ ... ] distinguish
a correctly-specified tail from an incorrect one?

Let me numerically verify the "tail CRPS" idea.
"""
import torch
import math

torch.manual_seed(42)
SQRT2 = math.sqrt(2.0); SQRTPI = math.sqrt(math.pi); SQRT2PI = math.sqrt(2.0*math.pi)

def gaussian_crps_correct(mu, sigma, y):
    z = (y - mu) / sigma
    Phi = 0.5 * (1 + torch.erf(z / SQRT2))
    phi = torch.exp(-0.5 * z * z) / SQRT2PI
    return sigma * (z * (2 * Phi - 1) + 2 * phi) - sigma / SQRTPI

def gaussian_tail_crps(mu, sigma, y, alpha):
    """CRPS of the alpha-tail-conditioned distribution, evaluated at y.
    Conditional CDF: F_T(x) = F(x) / alpha for x <= q_alpha, 0 otherwise.
    We use the identity  CRPS(F, y) = E|X - y| - 0.5 E|X - X'|.
    Conditioning X on the tail {X <= q}:
        E[X | X<=q, y] etc. For Gaussian we have closed form via truncated normal.
    Instead of deriving closed form, use MC samples of the truncated Gaussian.
    """
    q = mu + sigma * torch.special.ndtri(torch.tensor(alpha))
    # truncated normal samples below q
    # Use inverse CDF: u ~ Uniform(0, alpha), x = F^{-1}(u) = mu + sigma * ndtri(u)
    n_samp = 500
    u = torch.rand(y.shape[0], n_samp) * alpha
    x = mu[:, None] + sigma[:, None] * torch.special.ndtri(u)
    # CRPS via fair MC estimator
    a = (x - y[:, None]).abs().mean(1)
    # fair term: E|X - X'| for X,X' iid from the conditional
    x2 = mu[:, None] + sigma[:, None] * torch.special.ndtri(torch.rand(y.shape[0], n_samp) * alpha)
    b = (x - x2).abs().mean(1)
    return a - 0.5 * b

# Test: y ~ N(0,1) truly, restricted to y below q_true = ndtri(0.05) = -1.645.
y_all = torch.randn(200000)
q_true = torch.special.ndtri(torch.tensor(0.05)).item()
tail_mask = y_all <= q_true
y_tail = y_all[tail_mask]
print(f"tail sample size = {y_tail.shape[0]}, empirical tail prob = {tail_mask.float().mean():.4f} (want ~0.05)")

print("\nExpected tail-CRPS as a function of forecast (mu=0, sigma), on TRUE tail data")
print("(minimum should be at sigma=1 if proper for the tail shape)")
sigmas = torch.linspace(0.6, 2.0, 15)
for s in sigmas:
    mu_t = torch.zeros_like(y_tail)
    sigma_t = torch.full_like(y_tail, s.item())
    with torch.no_grad():
        v = gaussian_tail_crps(mu_t, sigma_t, y_tail, alpha=0.05).mean().item()
    bar = "#" * int(v * 50)
    print(f"  sigma={s:.3f}: {v:.5f} {bar}")
