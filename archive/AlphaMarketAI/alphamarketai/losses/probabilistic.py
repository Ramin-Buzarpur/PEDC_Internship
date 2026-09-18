import torch


def gaussian_nll(mu, sigma, target):
    return (torch.log(sigma) + 0.5 * ((target - mu) / sigma).square()).mean()


def crps_loss(samples, target):
    """CRPS for samples shaped [..., scenarios] and targets shaped [...]."""
    first = (samples - target.unsqueeze(-1)).abs().mean()
    second = 0.5 * (samples.unsqueeze(-1) - samples.unsqueeze(-2)).abs().mean()
    return first - second
