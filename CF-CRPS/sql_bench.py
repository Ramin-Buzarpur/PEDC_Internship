"""
SQL-CRPS v0.2 -- NN Benchmark on GARCH-like synthetic data
==========================================================
Backbone identical for all methods: MLP(20->64->32), heads mu/sigma/nu.
Head: Gaussian (for NLL/CRPS/SQL) vs Student-t (for NLL baseline).
Student-t is trained with NLL (analytic) -- the strongest baseline from the
user's v1.0 report.

Metrics: overall CRPS (MC on model samples), Tail-CRPS (|y| > q90 empirical),
VaR95/99 coverage (nominal 0.95/0.99 -- deviation from nominal is error),
ES (expected shortfall) gap, pit-tail calibration.

Methods:
  gaussian_nll
  studentt_nll
  gaussian_crps       (analytic)
  sql_lam=0.0         (= quantile-layer CRPS, should match analytic CRPS)
  sql_lam=0.5
  sql_lam=0.9
  sql_state           (lam depends on exogenous regime: lam_s = 0.9 * s)
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from sql_crps import make_alpha_grid, sql_crps_loss, gaussian_crps, \
    gaussian_nll, gaussian_quantiles, pinball

SQRT2 = math.sqrt(2.0)
SQRTPI = math.sqrt(math.pi)
SQRT2PI = math.sqrt(2.0 * math.pi)


# ---------------------------------------------------------------- data
def make_garch_data(n=6000, seed=0):
    """
    Regime-switching heteroscedastic returns with fat tails + crashes.
    x = 20 iid features; only regime signal is recoverable (leaks through
    x[0] and x[1]).
    """
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(n, 20, generator=g)
    regime = torch.rand(n, generator=g)
    sigma_true = 0.2 + 1.6 * regime
    y = torch.randn(n, generator=g) * sigma_true
    crash = torch.rand(n, generator=g) < (0.02 + 0.06 * regime)
    y[crash] -= torch.abs(torch.randn(crash.sum(), generator=g) * 4.0)
    # make regime recoverable from features
    x[:, 0] += 2.0 * regime
    x[:, 1] += 1.5 * (regime - 0.5)
    return x, y, regime


# ---------------------------------------------------------------- model
class ForecastNet(nn.Module):
    def __init__(self, d=20):
        super().__init__()
        self.body = nn.Sequential(nn.Linear(d, 64), nn.ReLU(),
                                  nn.Linear(64, 32), nn.ReLU())
        self.mu = nn.Linear(32, 1)
        self.sigma = nn.Linear(32, 1)
        self.nu_raw = nn.Parameter(torch.tensor(1.5))

    def forward(self, x):
        h = self.body(x)
        mu = self.mu(h).squeeze(-1)
        sigma = F.softplus(self.sigma(h).squeeze(-1)) + 1e-4
        nu = 2.1 + F.softplus(self.nu_raw)
        return mu, sigma, nu


def student_nll(y, mu, sigma, nu):
    z = (y - mu) / sigma
    logp = (torch.lgamma((nu + 1) / 2) - torch.lgamma(nu / 2)
            - 0.5 * torch.log(nu * math.pi) - torch.log(sigma)
            - ((nu + 1) / 2) * torch.log1p(z * z / nu))
    return -logp


# ---------------------------------------------------------------- loss
ALPHAS, WEIGHTS = make_alpha_grid()


def loss_fn(method, model, x, y, regime):
    mu, sigma, nu = model(x)
    if method == "gaussian_nll":
        return gaussian_nll(mu, sigma, y).mean()
    if method == "studentt_nll":
        return student_nll(y, mu, sigma, nu).mean()
    if method == "gaussian_crps":
        return gaussian_crps(mu, sigma, y).mean()
    if method.startswith("sql"):
        if method == "sql_state":
            # state-adaptive: high regime -> more tail emphasis
            lam = 0.9 * regime
        else:
            lam = float(method.split("=")[1])
        return sql_crps_loss(mu, sigma, y, ALPHAS, WEIGHTS,
                             lam=lam, alpha0=0.05).mean()
    raise ValueError(method)


# ---------------------------------------------------------------- eval
def mc_crps_from_samples(samples, y):
    a = (samples - y[None, :]).abs().mean(0)
    h = samples.shape[0] // 2
    b = (samples[:h] - samples[h:2 * h]).abs().mean(0)
    return a - 0.5 * b


def evaluate(model, x, y, seed=123):
    with torch.no_grad():
        mu, sigma, nu = model(x)
        # Gaussian MC samples for CRPS (same evaluator for all methods --
        # Student-t predictions use t-samples with its OWN nu)
        g = torch.Generator().manual_seed(seed)
        # Student-t sampling: t = Z / sqrt(chi2/nu); gives t-samples for the
        # Student-t head and standard-normal-equivalent samples when nu is
        # huge (not the case here, but same evaluator code path for all).
        z0 = torch.randn((500, x.shape[0]), generator=g)
        chi2 = torch.distributions.Chi2(nu).rsample((500, x.shape[0]))
        z = z0 / torch.sqrt(chi2 / nu[None])
        samples = mu[None] + sigma[None] * z
        crps = mc_crps_from_samples(samples, y)

        tail_mask = y.abs() > torch.quantile(y.abs(), 0.9)
        tail_crps = crps[tail_mask].mean()

        q95 = torch.quantile(samples, 0.05, dim=0)
        q99 = torch.quantile(samples, 0.01, dim=0)
        var95 = (y >= q95).float().mean().item()
        var99 = (y >= q99).float().mean().item()

        # Expected shortfall gap: E[y | y <= q05_true] vs model ES
        es_lvl = 0.05
        true_es = y[y <= torch.quantile(y, es_lvl)].mean()
        model_es = samples[samples <= q95[None]].mean()
        es_gap = abs(model_es.item() - true_es.item())

        return {"CRPS": crps.mean().item(),
                "TailCRPS": tail_crps.item(),
                "VaR95": var95, "VaR99": var99, "ESgap": es_gap}


# ---------------------------------------------------------------- run
def run(method, seed, epochs=120, lr=3e-3, n=6000):
    torch.manual_seed(seed)
    x, y, regime = make_garch_data(n, seed)
    model = ForecastNet()
    opt = torch.optim.Adam(model.parameters(), lr=lr)

    for ep in range(epochs):
        loss = loss_fn(method, model, x, y, regime)
        opt.zero_grad(); loss.backward(); opt.step()
        if not torch.isfinite(loss):
            return {k: float("nan") for k in
                    ["CRPS", "TailCRPS", "VaR95", "VaR99", "ESgap"]}

    return evaluate(model, x, y)


if __name__ == "__main__":
    seeds = [1, 2, 3, 42, 777]
    methods = ["gaussian_nll", "studentt_nll", "gaussian_crps",
               "sql_lam=0.0", "sql_lam=0.5", "sql_lam=0.9", "sql_state"]
    results = {}
    for m in methods:
        rows = [run(m, s) for s in seeds]
        results[m] = {k: sum(r[k] for r in rows) / len(rows)
                      for k in rows[0]}
        print(f"{m:15s} " + "  ".join(
            f"{k}={results[m][k]:.4f}" for k in rows[0]), flush=True)

    print("\n=== summary (mean over 5 seeds) ===")
    hdr = ["method", "CRPS", "TailCRPS", "VaR95", "VaR99", "ESgap"]
    print(("%-15s" + "%9s" * 5) % tuple(hdr))
    for m in methods:
        r = results[m]
        print(("%-15s" + "%9.4f" * 5) % ((m,) + tuple(
            r[k] for k in ["CRPS", "TailCRPS", "VaR95", "VaR99", "ESgap"])))
