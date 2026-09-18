"""
Diag 8: compare two MC estimators of the CRPS fair term.
(A) independent pairing  E|X1 - X2|  with fresh x1, x2
(B) sorted/sorted pairing or (n-1) normalized V-statistic  E|X - X'| over all pairs

Compute both against analytic CRPS to see which is correct.
"""
import torch
import math

torch.manual_seed(7)
SQRT2 = math.sqrt(2.0); SQRT2PI = math.sqrt(2.0 * math.pi)

def crps_analytic(mu, sigma, y):
    z = (y - mu) / sigma
    Phi = 0.5 * (1 + torch.erf(z / SQRT2))
    phi = torch.exp(-0.5 * z * z) / SQRT2PI
    return sigma * (z * (2 * Phi - 1) + 2 * phi - 1 / SQRT2PI)

y = torch.randn(2000)
for s in [0.5, 1.0, 2.0]:
    n_mc = 5000
    x1 = torch.randn(n_mc, 2000) * s
    x2 = torch.randn(n_mc, 2000) * s

    # (A) independent pairing
    est_A = (x1 - y[None, :]).abs().mean(0) - 0.5 * (x1 - x2).abs().mean(0)

    # (B) V-statistic: all ordered pairs including self-pairs
    # E|X-X'| over all n^2 pairs = (2/(n^2)) * sum_{i<j} |xi-xj| (for iid)
    # Approximate by permutation estimator: shuffle x1 and pair with x1
    perm = torch.randperm(n_mc)
    est_B = (x1 - y[None, :]).abs().mean(0) - 0.5 * (x1 - x1[perm]).abs().mean(0)

    an = crps_analytic(torch.zeros_like(y), torch.full_like(y, s), y)
    print(f"sigma={s}: analytic={an.mean().item():.4f}  A(indep)={est_A.mean().item():.4f}  B(perm)={est_B.mean().item():.4f}")
