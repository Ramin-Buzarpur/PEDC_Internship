"""
SQL-CRPS v0.3 -- full pipeline on SAFPS-style synthetic market
==============================================================
Data: the user's make_synthetic_data (heteroskedastic t4 noise + state-
dependent downside exponential shocks). Backbones identical.

Key v0.3 innovation being tested: the loss is BUILT IN TWO STAGES.
  Stage 1 (probability model): Gaussian NLL -- maximally stable mu/sigma.
  Stage 2 (tail distribution shape): train a SMALL nu-head with SQL-CRPS
  through a Student-t quantile layer while mu/sigma stay FROZEN.
Rationale: under the quantile-layer view, the tail layers alpha < alpha0
respond to the CDF shape in the far tail, which is controlled by nu. Training
nu with SQL's tail-concentrated weights (and mu/sigma frozen, protecting the
core calibration) targets tail metrics without sacrificing overall CRPS --
exactly the trade that killed earlier single-loss attempts.
"""
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from sql_crps import make_alpha_grid, gaussian_crps, gaussian_nll

SQRT2 = math.sqrt(2.0)
SQRTPI = math.sqrt(math.pi)
SQRT2PI = math.sqrt(2.0 * math.pi)


# ------------------------------------------------------------------ data
def make_synthetic_data(n, seed):
    """User's SAFPS generator (numpy), ported verbatim."""
    rng = np.random.default_rng(seed)
    signal = rng.normal(size=n)
    risk_state = rng.beta(2.0, 3.0, size=n)
    dir_unc = rng.beta(2.0, 2.0, size=n)
    mu = 0.8 * np.tanh(signal) * (1.0 - 0.7 * dir_unc)
    sigma = 0.55 + 1.15 * risk_state
    eps = rng.standard_t(df=4, size=n) / np.sqrt(2.0)
    y = mu + sigma * eps
    p_tail = 0.02 + 0.25 * risk_state**2
    shock = rng.random(n) < p_tail
    shock_size = 2.4 + 2.1 * risk_state[shock] + rng.exponential(0.8,
                                                                size=shock.sum())
    y[shock] -= shock_size
    X = np.stack([signal, risk_state, dir_unc], axis=1).astype(np.float32)
    return X, y.astype(np.float32)


# ------------------------------------------------------------------ model
class GaussianNet(nn.Module):
    """Standard backbone: predicts mu, sigma. Optionally freezes sigma."""

    def __init__(self, d=3, frozen=False):
        super().__init__()
        self.body = nn.Sequential(nn.Linear(d, 16), nn.Tanh(),
                                  nn.Linear(16, 16), nn.Tanh())
        self.mu_head = nn.Linear(16, 1)
        self.sigma_head = nn.Linear(16, 1)
        self.frozen = frozen
        if frozen:   # freeze everything except nu-head
            for p in self.body.parameters():
                p.requires_grad_(False)
            for p in self.mu_head.parameters():
                p.requires_grad_(False)
            for p in self.sigma_head.parameters():
                p.requires_grad_(False)

        self.nu_raw = nn.Parameter(torch.tensor(1.2))

    def forward(self, x):
        h = self.body(x)
        mu = self.mu_head(h).squeeze(-1)
        sigma = F.softplus(self.sigma_head(h).squeeze(-1)) + 1e-4
        nu = 2.1 + F.softplus(self.nu_raw)
        return mu, sigma, nu


# ---------------------------------------------------------- SQL machinery
ALPHAS, WEIGHTS = make_alpha_grid()
# small alpha floor for numerical stability of t-quantiles
ALPHAS = ALPHAS.clamp(min=2e-3)


def studentt_quantiles(mu, sigma, nu, alphas):
    """Student-t quantiles via vectorized Newton inversion of the t-CDF.

    t-CDF (for q > 0): F(q) = 1 - 0.5 * I_{nu/(nu+q^2)}(nu/2, 1/2)
    Newton iterations from a normal-quantile init; 25 steps converge to
    ~1e-6 for nu >= 2 over alpha in [2e-3, 1-2e-3]. The iteration is
    differentiable (implicit function theorem), so gradients flow to
    (mu, sigma, nu).
    """
    x = alphas[None, :].expand(mu.shape[0], -1)      # [n, K]
    sign = torch.where(x < 0.5, -torch.ones_like(x), torch.ones_like(x))
    p = torch.where(x < 0.5, x, 1 - x)               # lower-tail prob

    q = torch.special.ndtri(p)                       # init: normal quantile
    nu2 = nu[:, None].expand_as(q).contiguous()
    for _ in range(25):
        qq = q.abs()
        zz = nu2 / (nu2 + qq * qq)
        ib = _ibeta(zz, nu2 / 2, 0.5)
        cdf_pos = 1 - 0.5 * ib                       # CDF at +qq
        # CDF at signed q: F(-a) = 1 - F(a)
        cdf = torch.where(q > 0, cdf_pos, 1 - cdf_pos)
        pdf = _t_pdf(q, nu2)
        q = q - (cdf - p) / (pdf + 1e-12)
    q = sign * q.abs()
    return mu[:, None] + sigma[:, None] * q


def _ibeta(z, a, b):
    """Delegate to the dedicated differentiable implementation."""
    from math_utils import ibeta
    return ibeta(z, a, b)



def _t_pdf(q, nu):
    """Student-t pdf, vectorized: nu is a tensor broadcastable to q."""
    log_c = (torch.lgamma((nu + 1) / 2) - torch.lgamma(nu / 2)
             - 0.5 * torch.log(nu * math.pi))
    return torch.exp(log_c - ((nu + 1) / 2) * torch.log1p(q * q / nu))


# ---------------------------------------------------------------- losses
def student_nll(y, mu, sigma, nu):
    z = (y - mu) / sigma
    logp = (torch.lgamma((nu + 1) / 2) - torch.lgamma(nu / 2)
            - 0.5 * torch.log(nu * math.pi) - torch.log(sigma)
            - ((nu + 1) / 2) * torch.log1p(z * z / nu))
    return -logp


def sql_loss(method, mu, sigma, nu, y, regime):
    if method.startswith("sql"):
        if method == "sql_state":
            lam = 0.9 * regime
        else:
            lam = float(method.split("=")[1])
        if nu.dim() == 0:                       # scalar nu -> expand
            nu = nu.expand_as(mu)
        q = studentt_quantiles(mu, sigma, nu, ALPHAS)
        rho = ((y[:, None] - q) * (ALPHAS - (y[:, None] < q).float()))
        w_eff = ((1 - (lam[:, None] if isinstance(lam, torch.Tensor) else lam))
                 + (lam if isinstance(lam, torch.Tensor) else lam)
                 * (ALPHAS <= 0.05).float() / 0.05) * WEIGHTS
        w_eff = w_eff / w_eff.sum(-1, keepdim=True)
        return 2.0 * (w_eff * rho).sum(-1)
    raise ValueError(method)


# ---------------------------------------------------------------- training
def train(method, seed, epochs=120, lr=3e-3, n=7000):
    torch.manual_seed(seed)
    X, y_np = make_synthetic_data(n, seed)
    x = torch.from_numpy(X)
    y = torch.from_numpy(y_np)
    regime = x[:, 1].clamp(0, 1)   # risk_state

    model = GaussianNet()
    opt = torch.optim.Adam(model.parameters(), lr=lr)

    if method == "gaussian_nll":
        for ep in range(epochs):
            mu, sigma, nu = model(x)
            loss = gaussian_nll(mu, sigma, y).mean()
            opt.zero_grad(); loss.backward(); opt.step()

    elif method == "studentt_nll":
        for ep in range(epochs):
            mu, sigma, nu = model(x)
            loss = student_nll(y, mu, sigma, nu).mean()
            opt.zero_grad(); loss.backward(); opt.step()

    elif method == "gaussian_crps":
        for ep in range(epochs):
            mu, sigma, nu = model(x)
            loss = gaussian_crps(mu, sigma, y).mean()
            opt.zero_grad(); loss.backward(); opt.step()

    elif method == "sql_two_stage":
        # stage 1: gaussian NLL (probability model)
        for ep in range(epochs):
            mu, sigma, nu = model(x)
            loss = gaussian_nll(mu, sigma, y).mean()
            opt.zero_grad(); loss.backward(); opt.step()
        # stage 2: freeze mu/sigma path; train nu-head with SQL tail weight
        model2 = GaussianNet(frozen=True)
        model2.body.load_state_dict(model.body.state_dict())
        model2.mu_head.load_state_dict(model.mu_head.state_dict())
        model2.sigma_head.load_state_dict(model.sigma_head.state_dict())
        model2.nu_raw = model.nu_raw
        opt2 = torch.optim.Adam([model2.nu_raw], lr=5e-3)
        for ep in range(epochs):
            mu, sigma, nu = model2(x)
            loss = sql_loss("sql_lam=0.9", mu, sigma, nu, y, regime).mean()
            opt2.zero_grad(); loss.backward(); opt2.step()
        model = model2

    else:
        for ep in range(epochs):
            mu, sigma, nu = model(x)
            loss = sql_loss(method, mu, sigma, nu, y, regime).mean()
            opt.zero_grad(); loss.backward(); opt.step()

    return model, x, y, regime


def evaluate(model, x, y, seed=123):
    with torch.no_grad():
        mu, sigma, nu = model(x)
        g = torch.Generator().manual_seed(seed)
        z0 = torch.randn((500, x.shape[0]), generator=g)
        chi2 = torch.distributions.Chi2(nu.clamp(min=2.05)).rsample(
            (500, x.shape[0]))
        z = z0 / torch.sqrt(chi2 / nu.clamp(min=2.05)[None])
        samples = mu[None] + sigma[None] * z

        a = (samples - y[None, :]).abs().mean(0)
        h = 250
        b = (samples[:h] - samples[h:2 * h]).abs().mean(0)
        crps = a - 0.5 * b

        tail_mask = y.abs() > torch.quantile(y.abs(), 0.9)
        q95 = torch.quantile(samples, 0.05, dim=0)
        q99 = torch.quantile(samples, 0.01, dim=0)
        true_es = y[y <= torch.quantile(y, 0.05)].mean()
        model_es = samples[samples <= q95[None]].mean()
        return {"CRPS": crps.mean().item(),
                "TailCRPS": crps[tail_mask].mean().item(),
                "VaR95": (y >= q95).float().mean().item(),
                "VaR99": (y >= q99).float().mean().item(),
                "ESgap": abs(model_es.item() - true_es.item())}


if __name__ == "__main__":
    seeds = [1, 2, 3, 42, 777]
    methods = ["gaussian_nll", "studentt_nll", "gaussian_crps",
               "sql_lam=0.0", "sql_lam=0.5", "sql_lam=0.9",
               "sql_state", "sql_two_stage"]
    results = {}
    for m in methods:
        rows = []
        for s in seeds:
            model, x, y, regime = train(m, s)
            rows.append(evaluate(model, x, y))
        results[m] = {k: sum(r[k] for r in rows) / len(rows) for k in rows[0]}
        print(f"{m:15s} " + "  ".join(
            f"{k}={results[m][k]:.4f}" for k in rows[0]), flush=True)

    print("\n=== summary (mean over 5 seeds, SAFPS-style data) ===")
    print(("%-15s" + "%9s" * 5) % ("method", "CRPS", "TailCRPS", "VaR95",
                                   "VaR99", "ESgap"))
    for m in methods:
        r = results[m]
        print(("%-15s" + "%9.4f" * 5) % ((m,) + tuple(
            r[k] for k in ["CRPS", "TailCRPS", "VaR95", "VaR99", "ESgap"])))
