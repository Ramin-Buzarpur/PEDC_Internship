from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn


class ConditionalDiffusion:
    def __init__(
        self,
        model: nn.Module,
        num_timesteps: int = 60,
        beta_start: float = 1e-4,
        beta_end: float = 0.2,
        future_weight: float = 1.0,
        past_weight: float = 0.5,
        seq_len_future: int = 10,
        device: torch.device | str = "cpu",
    ):
        self.model = model.to(device)
        self.K = num_timesteps
        self.future_weight = future_weight
        self.past_weight = past_weight
        self.H_future = seq_len_future
        self.device = torch.device(device)
        betas = torch.linspace(beta_start, beta_end, num_timesteps, device=self.device, dtype=torch.float64)
        self.betas = betas.float()
        self.alphas = (1.0 - betas).float()
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)

    def q_sample(self, x0: torch.Tensor, t: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
        ac = self.alphas_cumprod[t].view(-1, 1, 1, 1)
        return ac.sqrt() * x0 + (1.0 - ac).sqrt() * noise

    def loss(self, x_all: torch.Tensor, cond: torch.Tensor, rel: torch.Tensor, sample_weight: torch.Tensor | None = None):
        B, C, N, T = x_all.shape
        t = torch.randint(0, self.K, (B,), device=x_all.device)
        noise = torch.randn_like(x_all)
        x_noisy = self.q_sample(x_all, t, noise)
        pred = self.model(x_noisy, t, cond, rel)
        pos_w = torch.full_like(x_all, self.past_weight)
        pos_w[..., -self.H_future :] = self.future_weight
        mse = ((noise - pred) ** 2) * pos_w
        if sample_weight is not None:
            w = sample_weight.view(B, 1, 1, 1).to(x_all.device, x_all.dtype)
            mse = mse * w
        return mse.mean()

    @torch.no_grad()
    def ddim_sample(self, condition: torch.Tensor, rel: torch.Tensor, steps: int = 15, eta: float = 0.0) -> torch.Tensor:
        B, C, N, T = condition.shape
        tau = np.linspace(self.K - 1, 0, steps + 1).round().astype(np.int64)
        x = torch.randn(B, C, N, T, device=condition.device)
        for i in range(steps):
            ti, tj = int(tau[i]), int(tau[i + 1])
            t_batch = torch.full((B,), ti, device=condition.device, dtype=torch.long)
            eps = self.model(x, t_batch, condition, rel)
            ac_i = self.alphas_cumprod[ti]
            x0 = (x - (1.0 - ac_i).sqrt() * eps) / ac_i.sqrt()
            x0 = x0.clamp(-8.0, 8.0)
            if tj <= 0:
                x = x0
                break
            ac_j = self.alphas_cumprod[tj]
            sigma = eta * ((1.0 - ac_j) / (1.0 - ac_i)).sqrt() * (1.0 - ac_i / ac_j).sqrt()
            dir_xt = (1.0 - ac_j - sigma**2).sqrt() * eps
            noise = torch.randn_like(x) if eta > 0 else torch.zeros_like(x)
            x = ac_j.sqrt() * x0 + dir_xt + sigma * noise
        return x

    @torch.no_grad()
    def ddpm_sample(self, condition: torch.Tensor, rel: torch.Tensor) -> torch.Tensor:
        B, C, N, T = condition.shape
        x = torch.randn(B, C, N, T, device=condition.device)
        for i in reversed(range(self.K)):
            t_batch = torch.full((B,), i, device=condition.device, dtype=torch.long)
            eps = self.model(x, t_batch, condition, rel)
            alpha = self.alphas[i]
            ac = self.alphas_cumprod[i]
            mean = (1.0 / alpha.sqrt()) * (x - ((1.0 - alpha) / (1.0 - ac).sqrt()) * eps)
            if i > 0:
                var = self.betas[i]
                x = mean + var.sqrt() * torch.randn_like(x)
            else:
                x = mean
        return x
