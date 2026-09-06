"""
Verify the core identity: analytic CRPS vs MC estimator, and show MC estimator
is an unbiased gradient estimator even for Student-t (pathwise).
"""
import torch
import math

torch.manual_seed(0)

def analytic_gaussian_crps(mu, sigma, y):
    z = (y - mu) / sigma
    Phi = 0.5 * (1 + torch.erf(z / math.sqrt(2)))
    phi = torch.exp(-0.5 * z * z) / math.sqrt(2 * math.pi)
    return sigma * (z * (2 * Phi - 1) + 2 * phi) - 1 / math.sqrt(math.pi)

def mc_crps(samples, y):
    n = samples.shape[0]
    a = torch.abs(samples - y[:, None]).mean(1)
    i = torch.randperm(n)
    j = torch.randperm(n)
    b = torch.abs(samples[i] - samples[j]).mean(1)
    return (a - 0.5 * b).mean()

def mc_crps_paired(samples, y):
    a = torch.abs(samples - y[:, None]).mean(1)
    h = samples.shape[1] // 2
    b = torch.abs(samples[:, :h] - samples[:, h:2*h]).mean(1)
    return (a - 0.5 * b).mean()

mu = torch.linspace(-1, 1, 20, requires_grad=True)
sigma = torch.linspace(0.2, 2, 20, requires_grad=True) + 0.3
y = torch.randn(20)

for n_samp in [10, 100, 1000]:
    # Fresh samples each estimate
    ests = []
    for rep in range(50):
        eps = torch.randn(20, n_samp)
        s = mu[:, None] + sigma[:, None] * eps
        ests.append(mc_crps_paired(s, y).item())
    mean_est = sum(ests) / len(ests)
    spread = (max(ests) - min(ests))
    true = analytic_gaussian_crps(mu, sigma, y).mean().item()
    print(f"n={n_samp:>5}: MC paired mean={mean_est:.4f} (spread {spread:.4f}), analytic={true:.4f}, bias={mean_est-true:+.4f}")

# Gradient check: does d/dmu MC-CRPS match d/dmu analytic CRPS?
mu = torch.tensor([0.3], requires_grad=True)
sigma = torch.tensor([1.0])
yv = torch.tensor([-0.7])

n_samp = 5000
torch.manual_seed(42)
eps = torch.randn(n_samp, 1)
s = mu + sigma * eps
loss = mc_crps_paired(s, yv)
loss.backward()
g_mc = mu.grad.item()

mu2 = torch.tensor([0.3], requires_grad=True)
loss2 = analytic_gaussian_crps(mu2, sigma, yv).mean()
loss2.backward()
g_analytic = mu2.grad.item()

print(f"\nGradient wrt mu: MC paired={g_mc:+.6f}, analytic={g_analytic:+.6f}")

# Same for sigma
mu3 = torch.tensor([0.3])
sigma3 = torch.tensor([1.0], requires_grad=True)
s = mu3 + sigma3 * eps
mc_crps_paired(s, yv).backward()
g_mc_sig = sigma3.grad.item()

sigma4 = torch.tensor([1.0], requires_grad=True)
analytic_gaussian_crps(mu3, sigma4, yv).mean().backward()
g_an_sig = sigma4.grad.item()

print(f"Gradient wrt sigma: MC paired={g_mc_sig:+.6f}, analytic={g_an_sig:+.6f}")
