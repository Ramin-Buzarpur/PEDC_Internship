"""
Diag 15b: fix NaN issue in tail-CRPS -- torch.special.ndtri(0) = -inf, so
clipping u away from 0. Also verify the argmin over sigma.
"""
import torch
import math

torch.manual_seed(42)
SQRT2 = math.sqrt(2.0); SQRTPI = math.sqrt(math.pi); SQRT2PI = math.sqrt(2.0*math.pi)

def gaussian_tail_crps(mu, sigma, y, alpha, n_samp=2000):
    q = mu + sigma * torch.special.ndtri(torch.tensor(alpha))
    u_lo = 1e-6  # avoid ndtri(0) = -inf
    u = u_lo + (alpha - u_lo) * torch.rand(y.shape[0], n_samp)
    x = mu[:, None] + sigma[:, None] * torch.special.ndtri(u)
    x2 = mu[:, None] + sigma[:, None] * torch.special.ndtri(u_lo + (alpha - u_lo) * torch.rand(y.shape[0], n_samp))
    a = (x - y[:, None]).abs().mean(1)
    b = (x - x2).abs().mean(1)
    return a - 0.5 * b

y_all = torch.randn(200000)
q_true = torch.special.ndtri(torch.tensor(0.05)).item()
tail_mask = y_all <= q_true
y_tail = y_all[tail_mask]

print("Expected tail-CRPS vs sigma on TRUE tail data:")
sigmas = torch.linspace(0.6, 2.0, 15)
vals = []
for s in sigmas:
    mu_t = torch.zeros_like(y_tail)
    sigma_t = torch.full_like(y_tail, s.item())
    with torch.no_grad():
        v = gaussian_tail_crps(mu_t, sigma_t, y_tail, alpha=0.05).mean().item()
    vals.append(v)
for s, v in zip(sigmas, vals):
    bar = "#" * int((v - min(vals)) * 500 + 1)
    print(f"  sigma={s:.3f}: {v:.5f} {bar}")
best = sigmas[vals.index(min(vals))]
print(f"\nargmin sigma = {best:.3f} (want ~1.0)")
