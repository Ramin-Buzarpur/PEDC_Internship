"""
Diag 10: What is the actual minimizer of E[TCM term] over sigma?
Derivative: d/dsigma E[(q - y)_+^2] with q = sigma * z_alpha, y ~ N(0,1).
It should be minimized at sigma = 1 (the true sigma) IF the pinball-loss
argument is right. Test numerically.
"""
import torch
import math

torch.manual_seed(42)
y = torch.randn(200000)
alpha = 0.05
z = torch.special.ndtri(torch.tensor(alpha)).item()

# E[(q - y)_+^2] where q = mu + sigma*z_alpha, with mu=0.
# If F = true N(0,1), sigma=1 gives q = z_alpha.
# For sigma < 1, q > z_alpha (less extreme), so MORE y's fall below q -> more penalty.
# For sigma > 1, q < z_alpha (more extreme), FEWER y's fall below q -> LESS penalty.
# So E[(q-y)_+^2] is monotonically decreasing in sigma! It never comes back up.
# That's why sigma got pushed to 1.75+.

# The pinball loss (proper for quantile) is:
#   rho_alpha(q, y) = alpha*(q-y)*1{y<=q} + (1-alpha)*(y-q)*1{y>q}
# This IS minimized at the true quantile. Let's verify:
print("pinball loss at level alpha, E over y~N(0,1), as function of sigma:")
sigmas = torch.linspace(0.6, 1.6, 21)
vals = []
for s in sigmas:
    q = s * z
    with torch.no_grad():
        rho = alpha * torch.relu(q - y) + (1 - alpha) * torch.relu(y - q)
        # also the upper tail
        q_hi = -q
        rho_hi = (1 - alpha) * torch.relu(y - q_hi) + alpha * torch.relu(q_hi - y)
        v = (rho + rho_hi).mean().item()
    vals.append(v)
for s, v in zip(sigmas, vals):
    bar = "#" * int((v - min(vals)) * 1000 + 1)
    print(f"  sigma={s:.3f}: {v:.5f} {bar}")
best = sigmas[vals.index(min(vals))]
print(f"\nargmin of pinball pair = {best:.3f} (want ~1.0)")
