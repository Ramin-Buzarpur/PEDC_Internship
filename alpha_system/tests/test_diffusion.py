from __future__ import annotations

import numpy as np
import torch

from src.models.diffusion import ConditionalDiffusion
from src.models.matchs import MaTCHS


def make_toy(C=4, N=5, T=40, device="cpu"):
    torch.manual_seed(1)
    model = MaTCHS(num_channels=C, seq_len_total=T, d_channels=16, mrt_layers=1, heads_masked=1, heads_free=2, num_groups=2)
    diff = ConditionalDiffusion(model, num_timesteps=30, beta_start=1e-4, beta_end=0.05, seq_len_future=10, device=device)
    return diff


def test_q_sample_interpolates_to_noise():
    diff = make_toy()
    x0 = torch.zeros(2, 4, 5, 40)
    t = torch.tensor([29, 29])
    noise = torch.randn(2, 4, 5, 40)
    xt = diff.q_sample(x0, t, noise)
    ac = diff.alphas_cumprod[29]
    expected_var = 1 - ac
    assert abs(xt.var().item() - expected_var.item()) < 0.05


def test_loss_finite_and_backward():
    diff = make_toy()
    x = torch.randn(3, 4, 5, 40)
    cond = x.clone()
    cond[..., -10:] = 0
    rel = torch.ones(5, 5, 2)
    loss = diff.loss(x, cond, rel)
    assert torch.isfinite(loss)
    loss.backward()


def test_ddim_sampling_shape_and_clamp():
    diff = make_toy()
    cond = torch.randn(2, 4, 5, 40) * 3
    rel = torch.ones(5, 5, 2)
    out = diff.ddim_sample(cond, rel, steps=6)
    assert out.shape == cond.shape
    assert (out.abs() <= 8.0 + 1e-5).all()


def test_overfit_toy_batch():
    diff = make_toy()
    opt = torch.optim.AdamW(diff.model.parameters(), lr=3e-4)
    x = torch.randn(8, 4, 5, 40)
    cond = x.clone()
    cond[..., -10:] = 0
    rel = torch.ones(5, 5, 2)
    first = None
    for i in range(25):
        loss = diff.loss(x, cond, rel)
        if first is None:
            first = float(loss.item())
        opt.zero_grad()
        loss.backward()
        opt.step()
    last = float(loss.item())
    assert last < first * 0.9
