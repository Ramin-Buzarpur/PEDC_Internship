"""
Diag 11: pinball loss with FREE q, not q tied to sigma.
Sanity check that the pinball loss IS minimized at the true quantile when q is
free. If yes, the issue is the TIE q = mu + sigma*z_alpha.
"""
import torch
import math

torch.manual_seed(42)
y = torch.randn(200000)
alpha = 0.05
z = torch.special.ndtri(torch.tensor(alpha)).item()

# free q: minimize empirical pinball
q_lo = torch.tensor(0.0, requires_grad=True)
q_hi = torch.tensor(0.0, requires_grad=True)
opt = torch.optim.Adam([q_lo, q_hi], lr=0.05)
for step in range(500):
    rho_lo = alpha * torch.relu(q_lo - y) + (1 - alpha) * torch.relu(y - q_lo)
    rho_hi = (1 - alpha) * torch.relu(y - q_hi) + alpha * torch.relu(q_hi - y)
    loss = (rho_lo + rho_hi).mean()
    opt.zero_grad(); loss.backward(); opt.step()
print(f"Free-q pinball argmin: q_lo = {q_lo.item():.4f} (true quantile = {z:.4f})")
print(f"                       q_hi = {q_hi.item():.4f} (true = {-z:.4f})")

# Now: tied q = mu + sigma*z. For the TIED case the pinball loss as function
# of sigma has what derivative at sigma=1?
# d/dsigma rho = [alpha*1{y<=q} - (1-alpha)*1{y>q}] * z  (for the lower tail)
# At sigma=1 (q = z_alpha true), P(y<=q) = alpha, so:
#   E[d/dsigma rho] = [alpha*alpha - (1-alpha)*(1-alpha)] * z
#                   = (2*alpha - 1) * z
# For alpha=0.05: (2*0.05-1)*(-1.645) = (-0.9)*(-1.645) = +1.48
# POSITIVE, meaning increasing sigma INCREASES loss -> optimizer wants to
# DECREASE sigma. So tied pinball pushes sigma DOWN, not to truth!
# Check the upper tail too: by symmetry, derivative = -1.48, wants sigma UP.
# Wait, that contradicts the numeric. Let me just compute numerically.
print()
print("Numerical dE[rho]/dsigma at sigma=1 for tied q:")
for a in [0.05, 0.10, 0.25, 0.50]:
    z_a = torch.special.ndtri(torch.tensor(a)).item()
    # estimate E[d rho / d sigma] at sigma=1
    q = 1.0 * z_a
    d_lower = (a * (y <= q).float() - (1 - a) * (y > q).float()) * z_a
    q_hi = -q
    d_upper = ((1 - a) * (y >= q_hi).float() - a * (y < q_hi).float()) * (-z_a)
    d = (d_lower + d_upper).mean().item()
    print(f"  alpha={a}: dE[rho]/dsigma at sigma=1 is {d:+.4f} "
          f"({'pushes sigma UP' if d < 0 else 'pushes sigma DOWN'})")
