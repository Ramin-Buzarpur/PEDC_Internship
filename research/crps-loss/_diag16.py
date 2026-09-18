"""
Diag 16: Tail-CRPS with a HETEROSCEDASTIC truth -- the key finance setting.
Truth: sigma alternates between 0.5 (calm) and 2.0 (crisis) based on a
lagged observable state. Check that a single global sigma CANNOT fit both
tails, motivating the state-conditional loss.
"""
import torch
import math

torch.manual_seed(42)

def gaussian_crps_correct(mu, sigma, y):
    SQRT2 = math.sqrt(2.0); SQRTPI = math.sqrt(math.pi); SQRT2PI = math.sqrt(2.0*math.pi)
    z = (y - mu) / sigma
    Phi = 0.5 * (1 + torch.erf(z / SQRT2))
    phi = torch.exp(-0.5 * z * z) / SQRT2PI
    return sigma * (z * (2 * Phi - 1) + 2 * phi) - sigma / SQRTPI

def gaussian_tail_crps(mu, sigma, y, alpha, n_samp=2000):
    u_lo = 1e-6
    u = u_lo + (alpha - u_lo) * torch.rand(y.shape[0], n_samp)
    x = mu[:, None] + sigma[:, None] * torch.special.ndtri(u)
    x2 = mu[:, None] + sigma[:, None] * torch.special.ndtri(u_lo + (alpha - u_lo) * torch.rand(y.shape[0], n_samp))
    a = (x - y[:, None]).abs().mean(1)
    b = (x - x2).abs().mean(1)
    return a - 0.5 * b

n = 30000
regime = (torch.rand(n) > 0.5).float()   # observable state
true_sigma = torch.where(regime == 1, torch.tensor(2.0), torch.tensor(0.5))
y = torch.randn(n) * true_sigma

# Fit:  mu, log_sigma = a * regime + b   (linear state model)
# Loss: CRPS + lam * TAIL-CRPS  (only on tail observations)
for lam in [0.0, 1.0, 5.0]:
    a = torch.tensor(0.0, requires_grad=True)
    b = torch.tensor(0.0, requires_grad=True)
    opt = torch.optim.Adam([a, b], lr=0.05)
    alpha = 0.05
    for step in range(500):
        log_s = a * regime + b
        mu = torch.zeros_like(y)
        sigma = torch.exp(log_s)
        crps = gaussian_crps_correct(mu, sigma, y)
        # tail term: indicator that y fell below the FORECAST's 5% quantile
        q05 = mu + sigma * torch.special.ndtri(torch.tensor(alpha))
        in_tail = (y <= q05).float()
        tail_crps = gaussian_tail_crps(mu, sigma, y, alpha) * in_tail
        loss = (crps + lam * tail_crps).mean()
        opt.zero_grad(); loss.backward(); opt.step()
    sigma_calm = torch.exp(b).item()
    sigma_crisis = torch.exp(a + b).item()
    print(f"lam={lam}: fitted sigma_calm={sigma_calm:.4f} (true 0.5), sigma_crisis={sigma_crisis:.4f} (true 2.0)")
