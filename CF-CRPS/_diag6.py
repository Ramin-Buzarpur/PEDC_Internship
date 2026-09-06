"""
Diag 6: where should TCM push sigma, in theory?
If the true distribution is N(0,1) and we use q_0.05 = -1.645*sigma as the
VaR, then the exceedance penalty E[(q-y)_+^2]/2alpha is minimized at what
sigma? Compute the expected value of the TCM term alone.
"""
import torch
import math

torch.manual_seed(42)
y = torch.randn(200000)
alpha = 0.05
z = torch.special.ndtri(torch.tensor(alpha)).item()
print(f"z_alpha = {z:.4f}")

sigmas = torch.linspace(0.8, 2.0, 25)
vals = []
for s in sigmas:
    q = s * z
    with torch.no_grad():
        left = torch.nn.functional.relu(q - y) ** 2
        right = torch.nn.functional.relu(y - (-q)) ** 2
        tcm = (0.5 * left + 0.5 * right).mean() / alpha
    vals.append(tcm.item())

for s, v in zip(sigmas, vals):
    bar = "#" * int((v - min(vals)) * 200 + 1)
    print(f"  sigma={s:.3f}: TCM={v:.5f} {bar}")
best = sigmas[vals.index(min(vals))]
print(f"\nargmin of TCM term alone = {best:.3f}")

# What about the CRPS term alone?
from math import sqrt, pi, erf, exp
def crps_analytic(mu, sigma, y):
    z_ = (y - mu) / sigma
    Phi = 0.5 * (1 + torch.erf(z_ / SQRT2))
    phi = torch.exp(-0.5 * z_ * z_) / SQRT2PI
    return sigma * (z_ * (2 * Phi - 1) + 2 * phi - 1 / SQRT2PI)

SQRT2 = math.sqrt(2.0); SQRT2PI = math.sqrt(2.0 * math.pi)
vals_crps = []
for s in sigmas:
    with torch.no_grad():
        v = crps_analytic(torch.zeros_like(y), torch.full_like(y, s.item()), y).mean().item()
    vals_crps.append(v)
best_crps = sigmas[vals_crps.index(min(vals_crps))]
print(f"argmin of CRPS term alone = {best_crps:.3f} (want ~1.0)")
