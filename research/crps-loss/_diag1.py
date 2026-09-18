"""
DIAGNOSTIC RUN: SC-twCRPS v1.0 exactly as it currently exists, to confirm the
numbers in report.txt and pinpoint the failure mode.
"""
import sys
sys.path.insert(0, "/home/alien-hawk/Desktop/PEDC/CF-CRPS")

from SC_twCRPS import (
    ForecastNet, student_nll, student_samples, mc_crps,
    sc_twcrps_penalty, generate_data, evaluate, run
)

import torch

# Reproduce the numbers in report.txt exactly (5 seeds per method is slow,
# first just 3 seeds to confirm the shape of the result).
for method in ["StudentT", "SC_only", "Hybrid", "SC_NoState"]:
    results = []
    for seed in [1, 2, 3]:
        results.append(run(method, seed))
    print(f"\n{method}  (3 seeds)")
    for k in results[0]:
        print(f"  {k}: {sum(r[k] for r in results)/len(results):.6f}")

# Additional diagnostic: what does SC_only actually learn?
torch.manual_seed(1)
x, y, vol = generate_data(n=3000)
model = ForecastNet()
opt = torch.optim.Adam(model.parameters(), lr=1e-3)
for epoch in range(60):
    mu, sigma, nu = model(x)
    sc = sc_twcrps_penalty(mu, sigma, nu, y, vol, use_state=True)
    opt.zero_grad(); sc.backward(); opt.step()

with torch.no_grad():
    mu, sigma, nu = model(x)
    print("\nSC_only learned distribution stats (diagnostic):")
    print(f"  mu    : mean={mu.mean():+.4f} std={mu.std():.4f} range=[{mu.min():.3f},{mu.max():.3f}]")
    print(f"  sigma : mean={sigma.mean():.4f} std={sigma.std():.4f} range=[{sigma.min():.3f},{sigma.max():.3f}]")
    print(f"  nu    : {nu.mean():.3f}")
    print(f"  |y|   : mean={y.abs().mean():.4f}")
    # CRPS of a CONSTANT predictor y=0 vs SC_only
    samples = student_samples(mu, sigma, nu)
    zero = torch.zeros_like(y)
    print(f"  SC_only CRPS: {mc_crps(samples, y).item():.4f}")
    print(f"  zero-mean-point-mass CRPS: {(y.abs().mean()).item():.4f}  (baseline)")
