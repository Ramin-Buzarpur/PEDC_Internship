"""
Diag 5b: Where is the true minimum of E[TCM-CRPS] over sigma?
The grid above was too coarse and cut off at 1.3. Compute the expected loss
over a finer, wider sigma grid to find the actual argmin.
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

torch.manual_seed(42)
y = torch.randn(200000)

alpha = 0.05
lam = 1.0
sigmas = torch.linspace(0.7, 2.0, 27)
vals = []
for s in sigmas:
    sigma_t = torch.full_like(y, s.item())
    with torch.no_grad():
        v = (gaussian_crps_analytic(torch.zeros_like(y), sigma_t, y)
             + lam * tcm_term(torch.zeros_like(y), sigma_t, y, alpha)).mean().item()
    vals.append(v)

for s, v in zip(sigmas, vals):
    bar = "#" * int((v - min(vals)) * 500 + 1)
    print(f"  sigma={s:.3f}: {v:.5f} {bar}")
best = sigmas[vals.index(min(vals))]
print(f"\nargmin sigma = {best:.3f}")
