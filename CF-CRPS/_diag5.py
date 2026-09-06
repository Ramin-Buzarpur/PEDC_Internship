"""
TEST 3: Is TCM-CRPS *locally* proper (minimum of expected loss at truth)?
Compute expected loss over the TRUE distribution for a grid of sigma
(perturbations around the true value), and check that the minimum is exactly
at sigma* = 1.
"""
import torch
import math

SQRT2 = math.sqrt(2.0)
SQRT2PI = math.sqrt(2.0 * math.pi)

def gaussian_crps_analytic(mu, sigma, y):
    z = (y - mu) / sigma
    Phi = 0.5 * (1 + torch.erf(z / SQRT2))
    phi = torch.exp(-0.5 * z * z) / SQRT2PI
    return sigma * (z * (2 * Phi - 1) + 2 * phi - 1 / SQRT2PI)

def gaussian_quantile(mu, sigma, alpha):
    z = torch.special.ndtri(torch.tensor(alpha))
    return mu + sigma * z

def tcm_term(mu, sigma, y, alpha):
    q_lo = gaussian_quantile(mu, sigma, alpha)
    q_hi = gaussian_quantile(mu, sigma, 1 - alpha)
    left = torch.nn.functional.relu(q_lo - y) ** 2
    right = torch.nn.functional.relu(y - q_hi) ** 2
    return (0.5 * left + 0.5 * right) / alpha

def tcm_crps_loss(mu, sigma, y, alpha=0.05, lam=1.0):
    return (gaussian_crps_analytic(mu, sigma, y) + lam * tcm_term(mu, sigma, y, alpha)).mean()

torch.manual_seed(42)
y = torch.randn(200000)  # true distribution is N(0,1)
alphas = [0.05, 0.10]
lams = [0.5, 1.0, 2.0]

print("Expected TCM-CRPS over N(0,1), as a function of forecast sigma")
print("(minimum should be at sigma=1.0 for ALL lam>0 if proper)")
print()
for alpha in alphas:
    for lam in lams:
        sigmas = [0.7, 0.85, 0.95, 1.0, 1.05, 1.15, 1.3]
        vals = []
        for s in sigmas:
            sigma_t = torch.full_like(y, s)
            with torch.no_grad():
                v = tcm_crps_loss(torch.zeros_like(y), sigma_t, y, alpha=alpha, lam=lam).item()
            vals.append(v)
        best = sigmas[vals.index(min(vals))]
        # Print normalized to value at sigma=1
        rel = [v - vals[sigmas.index(1.0)] for v in vals]
        row = " ".join(f"{s:4.2f}:{r:+.5f}" for s, r in zip(sigmas, rel))
        print(f"  alpha={alpha} lam={lam}: argmin sigma = {best:.2f}   [{row}]")
    print()

# Also check: does minimizing CRPS alone find sigma=1?
print("Control: plain CRPS argmin:")
sigmas = [0.7, 0.85, 0.95, 1.0, 1.05, 1.15, 1.3]
vals = []
for s in sigmas:
    sigma_t = torch.full_like(y, s)
    with torch.no_grad():
        v = gaussian_crps_analytic(torch.zeros_like(y), sigma_t, y).mean().item()
    vals.append(v)
best = sigmas[vals.index(min(vals))]
print(f"  argmin sigma = {best:.2f}  (want 1.0)")
