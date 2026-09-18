import math

import torch
from torch import nn
import torch.nn.functional as F


class DiffusionDenoiser(nn.Module):
    def __init__(self, future_len: int, condition_dim: int, hidden: int = 64):
        super().__init__()
        self.time = nn.Sequential(nn.Linear(1, hidden), nn.SiLU(), nn.Linear(hidden, hidden))
        self.network = nn.Sequential(
            nn.Linear(future_len + condition_dim + hidden, hidden * 2), nn.SiLU(),
            nn.Linear(hidden * 2, hidden), nn.SiLU(), nn.Linear(hidden, future_len),
        )

    def forward(self, noisy_future, timestep, condition):
        scaled_time = timestep.float().unsqueeze(-1) / 1000.0
        return self.network(torch.cat([noisy_future, condition, self.time(scaled_time)], dim=-1))


class GaussianDiffusion(nn.Module):
    def __init__(self, denoiser: DiffusionDenoiser, steps: int = 20):
        super().__init__()
        self.denoiser = denoiser
        self.steps = steps
        betas = torch.linspace(1e-4, 0.02, steps)
        alphas = 1.0 - betas
        cumulative = torch.cumprod(alphas, dim=0)
        self.register_buffer("betas", betas)
        self.register_buffer("alphas", alphas)
        self.register_buffer("alpha_cumulative", cumulative)

    def training_loss(self, future, condition):
        batch = future.shape[0]
        timestep = torch.randint(0, self.steps, (batch,), device=future.device)
        noise = torch.randn_like(future)
        cumulative = self.alpha_cumulative[timestep].unsqueeze(-1)
        noisy = cumulative.sqrt() * future + (1.0 - cumulative).sqrt() * noise
        predicted_noise = self.denoiser(noisy, timestep, condition)
        return F.mse_loss(predicted_noise, noise)

    @torch.no_grad()
    def sample(self, condition, num_samples=32):
        batch, assets, condition_dim = condition.shape
        flat_condition = condition.reshape(batch * assets, condition_dim).repeat_interleave(num_samples, 0)
        x = torch.randn(flat_condition.shape[0], self.denoiser.network[-1].out_features, device=condition.device)
        for index in reversed(range(self.steps)):
            timestep = torch.full((x.shape[0],), index, device=x.device, dtype=torch.long)
            predicted_noise = self.denoiser(x, timestep, flat_condition)
            alpha = self.alphas[index]
            cumulative = self.alpha_cumulative[index]
            mean = (x - (1 - alpha) / torch.sqrt(1 - cumulative) * predicted_noise) / torch.sqrt(alpha)
            x = mean if index == 0 else mean + torch.sqrt(self.betas[index]) * torch.randn_like(x)
        return x.reshape(batch, assets, num_samples, -1)
