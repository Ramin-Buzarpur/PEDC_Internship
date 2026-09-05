"""
SC-twCRPS v0.1 Prototype
State-Conditional Threshold-Weighted CRPS

First experimental loss implementation.
"""

import torch


def gaussian_cdf(x, mu, sigma):
    z = (x - mu) / sigma
    return 0.5 * (1.0 + torch.erf(z / torch.sqrt(torch.tensor(2.0, device=x.device))))


def gaussian_crps(mu, sigma, y):
    sigma = torch.clamp(sigma, min=1e-6)
    z = (y - mu) / sigma

    pdf = torch.exp(-0.5 * z ** 2) / torch.sqrt(
        torch.tensor(2.0 * torch.pi, device=y.device)
    )
    cdf = gaussian_cdf(y, mu, sigma)

    crps = sigma * (
        z * (2 * cdf - 1)
        + 2 * pdf
        - 1 / torch.sqrt(torch.tensor(torch.pi, device=y.device))
    )

    return crps.mean()


def threshold_weight(thresholds, realized_vol, lam=1.0):
    """
    Weight depends only on historical/exogenous state.
    """
    return torch.exp(
        lam * realized_vol.unsqueeze(-1) * torch.abs(thresholds)
    )


def sc_twcrps(mu, sigma, y, realized_vol, thresholds, lam=1.0):
    """
    Approximation of:

    integral w_t(u)(F(u)-I(y<=u))^2 du
    """

    t = thresholds.unsqueeze(0)

    F = gaussian_cdf(
        t,
        mu.unsqueeze(-1),
        sigma.unsqueeze(-1)
    )

    indicator = (y.unsqueeze(-1) <= t).float()

    error = (F - indicator) ** 2

    weights = threshold_weight(
        thresholds,
        realized_vol,
        lam
    )

    return torch.trapz(
        error * weights,
        thresholds,
        dim=-1
    ).mean()


def SC_twCRPS_loss(
    mu,
    sigma,
    y,
    realized_vol,
    thresholds,
    lam=1.0,
    alpha=1.0,
):
    """
    Final loss:

    CRPS + alpha * SC-twCRPS
    """

    return (
        gaussian_crps(mu, sigma, y)
        +
        alpha * sc_twcrps(
            mu,
            sigma,
            y,
            realized_vol,
            thresholds,
            lam
        )
    )


if __name__ == "__main__":

    n = 64

    mu = torch.randn(n)
    sigma = torch.ones(n) * 0.5
    y = torch.randn(n)

    # Must be calculated from past observations only.
    realized_vol = torch.rand(n)

    thresholds = torch.linspace(-3, 3, 200)

    loss = SC_twCRPS_loss(
        mu,
        sigma,
        y,
        realized_vol,
        thresholds,
        lam=1.0,
        alpha=0.5
    )

    print("SC-twCRPS v0.1:", loss.item())
