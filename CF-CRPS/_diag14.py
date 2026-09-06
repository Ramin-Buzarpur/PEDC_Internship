"""
Diag 14: What does the pinball loss at TIED quantiles actually minimize?
I claimed it's a proper scoring rule for the quantile, but with the tie
q = mu + sigma*z_alpha, the joint loss over (mu, sigma) is NOT minimized at
the truth. Let's check: what (mu*, sigma*) minimizes E[rho_alpha(mu+sigma*z, y)]
over (mu, sigma)? If it's the true (mu*, sigma*) I was wrong; otherwise the
pinball argument doesn't hold for tied quantiles.

Do gradient descent from a few starting points.
"""
import torch
import math

torch.manual_seed(42)
y = torch.randn(200000)
alpha = 0.05
z = torch.special.ndtri(torch.tensor(alpha)).item()
z_hi = -z

for lam in [0.0, 1.0]:
    for s0 in [0.5, 1.5]:
        mu_p = torch.tensor(0.0, requires_grad=True)
        log_s = torch.tensor(math.log(s0), requires_grad=True)
        opt = torch.optim.Adam([mu_p, log_s], lr=0.05)
        for step in range(600):
            sigma = torch.exp(log_s)
            q_lo = mu_p + sigma * z
            q_hi = mu_p + sigma * z_hi
            rho = (alpha * torch.relu(q_lo - y) + (1 - alpha) * torch.relu(y - q_lo)
                   + (1 - alpha) * torch.relu(y - q_hi) + alpha * torch.relu(q_hi - y))
            loss = rho.mean() * (lam + 1.0)  # scale doesn't matter, just for lam=0 case
            opt.zero_grad(); loss.backward(); opt.step()
        print(f"lam={lam} sigma_init={s0}: tied-pinball argmin mu={mu_p.item():+.4f}, "
              f"sigma={torch.exp(log_s).item():.4f}")
print("(if tied pinball were proper for sigma, we'd see mu=0, sigma=1)")
