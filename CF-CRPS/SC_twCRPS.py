
"""
SC-twCRPS v0.2
Smooth Tail State-Conditional Threshold Weighted CRPS

Design goals:
- Properness preserved by using exogenous past volatility state.
- Tail weighting only activates in extreme standardized regions.
- Per-sample thresholds using forecast distribution scale.
- Stable bounded weights.
"""

import torch


def gaussian_cdf(x, mu, sigma):
    sigma = torch.clamp(sigma, min=1e-6)
    z = (x - mu) / sigma
    return 0.5 * (
        1.0 + torch.erf(z / torch.sqrt(torch.tensor(2.0, device=x.device)))
    )


def gaussian_crps(mu, sigma, y):
    sigma = torch.clamp(sigma, min=1e-6)

    z = (y - mu) / sigma

    pdf = torch.exp(-0.5 * z ** 2) / torch.sqrt(
        torch.tensor(2.0 * torch.pi, device=y.device)
    )

    cdf = gaussian_cdf(y, mu, sigma)

    crps = sigma * (
        z * (2.0 * cdf - 1.0)
        + 2.0 * pdf
        - 1.0 / torch.sqrt(torch.tensor(torch.pi, device=y.device))
    )

    return crps.mean()


def smooth_tail_weight(
    z_abs,
    realized_vol,
    lam=1.0,
    cutoff=1.5,
    sharpness=5.0,
):
    """
    Smooth tail activation.

    z_abs:
        standardized threshold distance |(u-mu)/sigma|

    realized_vol:
        exogenous state from past information only
    """

    vol = torch.clamp(realized_vol.detach(), 0.0, 3.0)

    gate = torch.sigmoid(
        sharpness * (z_abs - cutoff)
    )

    boost = torch.exp(
        torch.clamp(lam * vol.unsqueeze(-1), max=5.0)
    ) - 1.0

    return 1.0 + boost * gate


def sc_twcrps_v02(
    mu,
    sigma,
    y,
    realized_vol,
    z_grid,
    lam=1.0,
    alpha=1.0,
):
    """
    Single-integral SC-twCRPS.

    Thresholds are generated per sample:

        u = mu + sigma*z

    """

    sigma = torch.clamp(sigma, min=1e-6)

    z = z_grid.unsqueeze(0)

    thresholds = (
        mu.unsqueeze(-1)
        +
        sigma.unsqueeze(-1) * z
    )

    F = gaussian_cdf(
        thresholds,
        mu.unsqueeze(-1),
        sigma.unsqueeze(-1)
    )

    indicator = (
        y.unsqueeze(-1) <= thresholds
    ).float()

    error = (F - indicator) ** 2

    weights = smooth_tail_weight(
        torch.abs(z),
        realized_vol,
        lam=lam,
    )

    weighted_error = (
        1.0 + alpha * (weights - 1.0)
    ) * error

    return torch.trapz(
        weighted_error,
        z_grid,
        dim=-1
    ).mean()


def SC_twCRPS_v02_loss(
    mu,
    sigma,
    y,
    realized_vol,
    z_grid,
    lam=1.0,
    alpha=1.0,
):
    return sc_twcrps_v02(
        mu,
        sigma,
        y,
        realized_vol,
        z_grid,
        lam,
        alpha,
    )


if __name__ == "__main__":

    n = 64

    mu = torch.randn(n)
    sigma = torch.rand(n) + 0.2
    y = torch.randn(n)

    # Example only:
    # must be calculated from past observations
    realized_vol = torch.rand(n)

    z_grid = torch.linspace(-5, 5, 200)

    loss = SC_twCRPS_v02_loss(
        mu,
        sigma,
        y,
        realized_vol,
        z_grid,
        lam=1.0,
        alpha=0.5,
    )

    print("SC-twCRPS v0.2:", loss.item())

    # sanity check:
    zero_state_loss = SC_twCRPS_v02_loss(
        mu,
        sigma,
        y,
        torch.zeros_like(realized_vol),
        z_grid,
        lam=1.0,
        alpha=0.5,
    )

    print("Zero volatility state:", zero_state_loss.item())
