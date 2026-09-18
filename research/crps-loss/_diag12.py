"""
Diag 12: derivative of CRPS w.r.t. sigma at the truth, for Student-t and
Gaussian families. Check that dCRPS/dsigma at (mu*, sigma*) is 0 for BOTH
(correct CRPS), and compare against the pinned bug behavior.
"""
import torch
import math

torch.manual_seed(0)
SQRT2 = math.sqrt(2.0); SQRTPI = math.sqrt(math.pi); SQRT2PI = math.sqrt(2.0*math.pi)

def gaussian_crps_correct(mu, sigma, y):
    z = (y - mu) / sigma
    Phi = 0.5 * (1 + torch.erf(z / SQRT2))
    phi = torch.exp(-0.5 * z * z) / SQRT2PI
    return sigma * (z * (2 * Phi - 1) + 2 * phi) - sigma / SQRTPI

y = torch.randn(100000)

# sigma gradient at truth for correct CRPS
for s0 in [0.5, 1.0, 2.0]:
    mu_p = torch.tensor(0.0, requires_grad=True)
    log_s = torch.tensor(math.log(s0), requires_grad=True)
    sigma = torch.exp(log_s)
    loss = gaussian_crps_correct(mu_p, sigma, y).mean()
    loss.backward()
    print(f"correct CRPS: at sigma={s0}, dCRPS/dlog_sigma = {log_s.grad.item():+.4f}")

# Now derivative of the tied pinball loss: E over y, at sigma=1, d/dsigma
alpha = 0.05
z = torch.special.ndtri(torch.tensor(alpha)).item()
q = z
d_lower = (alpha * (y <= q).float() - (1 - alpha) * (y > q).float()) * z
d_upper = ((1 - alpha) * (y >= -q).float() - alpha * (y < -q).float()) * (-z)
print(f"\ntied pinball: dE[rho]/dsigma at sigma=1 = {(d_lower + d_upper).mean().item():+.4f}")
print("(nonzero -> the tied pinball term is NOT minimized at sigma*=1 by itself)")
