"""
Diag 7: Numerically verify the analytic Gaussian CRPS against the definition
CRPS(F, y) = E|X - y| - 0.5 E|X - X'| with X, X' ~ F iid, using Monte Carlo.
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

y = torch.randn(5000)
for s in [0.5, 1.0, 2.0]:
    n_mc = 20000
    x1 = torch.randn(n_mc, 1) * s
    x2 = torch.randn(n_mc, 1) * s
    mc = (x1 - y[None, :]).abs().mean(0) - 0.5 * (x1 - x2).abs().mean()
    an = crps_analytic(torch.zeros_like(y), torch.full_like(y, s), y)
    print(f"sigma={s}: MC={mc.mean().item():.4f}  analytic={an.mean().item():.4f}  diff={mc.mean().item()-an.mean().item():+.4f}")
