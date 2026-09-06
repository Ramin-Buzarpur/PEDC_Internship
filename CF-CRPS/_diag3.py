"""
Diag 3: Bias of the biased MC-CRPS estimator with FEW samples, and its effect
on optimizing sigma.

Key question: at n=100 samples (what SC_twCRPS.py uses), how badly is the
pairing estimator biased as a function of true sigma? If bias ~ -c*sigma, then
optimizing it shrinks sigma toward 0.
"""
import torch
import math

torch.manual_seed(0)

def analytic_gaussian_crps(mu, sigma, y):
    z = (y - mu) / sigma
    Phi = 0.5 * (1 + torch.erf(z / math.sqrt(2)))
    phi = torch.exp(-0.5 * z * z) / math.sqrt(2 * math.pi)
    return sigma * (z * (2 * Phi - 1) + 2 * phi) - 1 / math.sqrt(math.pi)

def mc_crps_paired(samples, y):
    a = torch.abs(samples - y[:, None]).mean(1)
    h = samples.shape[1] // 2
    b = torch.abs(samples[:, :h] - samples[:, h:2*h]).mean(1)
    return (a - 0.5 * b).mean()

y = torch.randn(2000)
print("sigma_true | analytic CRPS | biased-MC CRPS estimate (n=100) | bias")
for sigma_val in [0.3, 0.6, 1.0, 1.5, 2.5]:
    sigma = torch.full_like(y, sigma_val)
    true = analytic_gaussian_crps(torch.zeros_like(y), sigma, y).mean().item()
    ests = []
    for rep in range(30):
        eps = torch.randn(2000, 100)
        s = sigma[:, None] * eps  # mu=0
        ests.append(mc_crps_paired(s, y).mean().item())
    m = sum(ests) / len(ests)
    print(f"  {sigma_val:4.1f}     |   {true:.4f}      |  {m:.4f}  |  {m-true:+.4f}")
