"""
Diag 9: sanity check the analytic CRPS formula itself on a SINGLE value of y
with a straightforward MC integration over x.
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

# single y, single mu, sigma
mu, sigma, yv = 0.0, 1.0, 0.5

# MC estimate of E|X-y| for X~N(mu,sigma)
n = 2_000_000
x = torch.randn(n) * sigma + mu
E1 = (x - yv).abs().mean().item()
E2 = 0.5 * 2 * sigma / SQRT2PI  # E|X-X'| for X,X' iid N(0,sigma^2) = 2 sigma / sqrt(pi) ... wait
# E|X-X'| = sigma * 2/sqrt(pi) * sqrt(2) ... let's compute directly
x2 = torch.randn(n) * sigma + mu
E2_emp = (x - x2).abs().mean().item()
E2_theory = 2 * sigma / math.sqrt(math.pi)
print(f"E|X-y| (MC)  = {E1:.5f}")
print(f"E|X-X'| (MC) = {E2_emp:.5f}, theory {E2_theory:.5f}")
print(f"CRPS MC = {E1 - 0.5*E2_emp:.5f}")
print(f"CRPS analytic = {crps_analyric_val if False else crps_analytic(torch.tensor(mu), torch.tensor(sigma), torch.tensor(yv)).item():.5f}")
