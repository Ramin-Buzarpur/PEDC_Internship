"""
Diag 13: verify the correct formula's argmin over sigma via grid search
(replacing diag 5 that used the buggy formula).
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

def gaussian_quantile(mu, sigma, alpha):
    z = torch.special.ndtri(torch.tensor(alpha))
    return mu + sigma * z

def tcm_term(mu, sigma, y, alpha):
    q_lo = gaussian_quantile(mu, sigma, alpha)
    q_hi = gaussian_quantile(mu, sigma, 1 - alpha)
    left = torch.nn.functional.relu(q_lo - y) ** 2
    right = torch.nn.functional.relu(y - q_hi) ** 2
    return (0.5 * left + 0.5 * right) / alpha

y = torch.randn(200000)

alpha = 0.05
lam = 1.0
sigmas = torch.linspace(0.7, 2.0, 27)
vals = []
for s in sigmas:
    sigma_t = torch.full_like(y, s.item())
    with torch.no_grad():
        v = (gaussian_crps_correct(torch.zeros_like(y), sigma_t, y)
             + lam * tcm_term(torch.zeros_like(y), sigma_t, y, alpha)).mean().item()
    vals.append(v)

best = sigmas[vals.index(min(vals))]
print(f"argmin of TCM-CRPS (correct CRPS formula) = {best:.3f}  (want ~1.0)")
for s, v in zip(sigmas, vals):
    bar = "#" * int((v - min(vals)) * 1000 + 1)
    print(f"  sigma={s:.3f}: {v:.5f} {bar}")
