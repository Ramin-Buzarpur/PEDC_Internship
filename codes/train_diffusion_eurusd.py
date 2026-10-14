# -*- coding: utf-8 -*-
"""
train_diffusion_eurusd.py
==========================
Trains a lightweight conditional DDPM (denoising diffusion) model on the
hourly EUR/USD log-return series produced by download_eurusd_hourly.py,
then samples N future scenarios over a 3-day horizon.

Design (kept deliberately small — this is a research prototype, not a
production forecaster):
  - Input: hourly log returns, windowed into (context, horizon) pairs.
      context_len bars  -> encoded by a small 1D-CNN into a condition vector
      horizon_len bars  -> the sequence the diffusion model learns to denoise
  - horizon_len is derived from the data itself (median bars/day * 3), not
    hardcoded to 72, because forex hourly data has weekend gaps.
  - Standard DDPM: fixed linear beta schedule, model predicts the noise
    epsilon, trained with MSE (Ho et al. 2020). Conditioning via FiLM
    (feature-wise linear modulation) at every conv block.
  - Everything needed to resample later (context_len, horizon_len,
    normalization stats, bars/day) is saved inside the checkpoint so
    mod4.py does not need to duplicate any of this logic.

Usage:
    python train_diffusion_eurusd.py            # train + save checkpoint
    python train_diffusion_eurusd.py --sample    # load checkpoint, print a
                                                  # quick sanity sample
"""

from __future__ import annotations

import math
import os
import sys
from dataclasses import dataclass, asdict

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
HOURLY_CSV = os.path.join(DATA_DIR, "EUR_USD_hourly.csv")
CHECKPOINT_PATH = os.path.join(DATA_DIR, "diffusion_eurusd.pt")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ============================================================================
# 1. Data loading & windowing
# ============================================================================
def load_hourly_log_returns(csv_path: str = HOURLY_CSV) -> tuple[np.ndarray, float]:
    df = pd.read_csv(csv_path, index_col=0)
    df.index = pd.to_datetime(df.index, utc=True)  # mixed UTC offsets (DST) defeat plain parse_dates
    close = df["Close"].to_numpy(dtype=np.float64)
    log_ret = np.diff(np.log(close))

    bars_per_day = float(df.groupby(df.index.date).size().median())
    return log_ret, bars_per_day


def make_windows(log_ret: np.ndarray, context_len: int, horizon_len: int) -> tuple[np.ndarray, np.ndarray]:
    total_len = context_len + horizon_len
    n_windows = len(log_ret) - total_len + 1
    if n_windows <= 0:
        raise ValueError("Not enough data for the requested context/horizon length.")

    ctx = np.zeros((n_windows, context_len), dtype=np.float32)
    hor = np.zeros((n_windows, horizon_len), dtype=np.float32)
    for i in range(n_windows):
        ctx[i] = log_ret[i : i + context_len]
        hor[i] = log_ret[i + context_len : i + context_len + horizon_len]
    return ctx, hor


# ============================================================================
# 2. Model: tiny FiLM-conditioned 1D-CNN noise predictor
# ============================================================================
def sinusoidal_embedding(timesteps: torch.Tensor, dim: int) -> torch.Tensor:
    half = dim // 2
    freqs = torch.exp(-math.log(10000) * torch.arange(half, device=timesteps.device).float() / half)
    args = timesteps.float()[:, None] * freqs[None, :]
    return torch.cat([torch.sin(args), torch.cos(args)], dim=-1)


class ContextEncoder(nn.Module):
    def __init__(self, context_len: int, cond_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=5, padding=2),
            nn.SiLU(),
            nn.Conv1d(32, 32, kernel_size=5, padding=2, stride=2),
            nn.SiLU(),
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(32, cond_dim),
        )

    def forward(self, ctx: torch.Tensor) -> torch.Tensor:
        return self.net(ctx.unsqueeze(1))


class FiLMBlock(nn.Module):
    def __init__(self, channels: int, cond_dim: int):
        super().__init__()
        self.conv = nn.Conv1d(channels, channels, kernel_size=5, padding=2)
        self.norm = nn.GroupNorm(8, channels)
        self.film = nn.Linear(cond_dim, channels * 2)
        self.act = nn.SiLU()

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        h = self.conv(x)
        h = self.norm(h)
        gamma, beta = self.film(cond).chunk(2, dim=-1)
        h = h * (1 + gamma[:, :, None]) + beta[:, :, None]
        return x + self.act(h)


class DenoiseNet(nn.Module):
    def __init__(self, horizon_len: int, context_len: int, time_dim: int = 64, cond_dim: int = 64, channels: int = 64, n_blocks: int = 4):
        super().__init__()
        self.time_dim = time_dim
        self.ctx_encoder = ContextEncoder(context_len, cond_dim)
        self.time_mlp = nn.Sequential(nn.Linear(time_dim, cond_dim), nn.SiLU(), nn.Linear(cond_dim, cond_dim))
        self.cond_merge = nn.Linear(cond_dim * 2, cond_dim)

        self.in_proj = nn.Conv1d(1, channels, kernel_size=5, padding=2)
        self.blocks = nn.ModuleList([FiLMBlock(channels, cond_dim) for _ in range(n_blocks)])
        self.out_proj = nn.Conv1d(channels, 1, kernel_size=5, padding=2)

    def forward(self, x_noisy: torch.Tensor, t: torch.Tensor, ctx: torch.Tensor) -> torch.Tensor:
        t_emb = self.time_mlp(sinusoidal_embedding(t, self.time_dim))
        c_emb = self.ctx_encoder(ctx)
        cond = self.cond_merge(torch.cat([t_emb, c_emb], dim=-1))

        h = self.in_proj(x_noisy.unsqueeze(1))
        for block in self.blocks:
            h = block(h, cond)
        return self.out_proj(h).squeeze(1)


# ============================================================================
# 3. DDPM schedule + train/sample loops
# ============================================================================
@dataclass
class DiffusionConfig:
    context_len: int
    horizon_len: int
    bars_per_day: float
    n_timesteps: int = 200
    mean: float = 0.0
    std: float = 1.0


class GaussianDiffusion:
    def __init__(self, n_timesteps: int, device: torch.device):
        self.T = n_timesteps
        betas = torch.linspace(1e-4, 0.02, n_timesteps, device=device)
        alphas = 1.0 - betas
        self.alphas_cumprod = torch.cumprod(alphas, dim=0)
        self.betas = betas
        self.alphas = alphas

    def q_sample(self, x0: torch.Tensor, t: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
        sqrt_ac = self.alphas_cumprod[t].sqrt()[:, None]
        sqrt_1m_ac = (1 - self.alphas_cumprod[t]).sqrt()[:, None]
        return sqrt_ac * x0 + sqrt_1m_ac * noise

    @torch.no_grad()
    def p_sample_loop(self, model: nn.Module, ctx: torch.Tensor, horizon_len: int) -> torch.Tensor:
        device = ctx.device
        n = ctx.shape[0]
        x = torch.randn(n, horizon_len, device=device)
        for t_idx in reversed(range(self.T)):
            t = torch.full((n,), t_idx, device=device, dtype=torch.long)
            eps_pred = model(x, t, ctx)
            alpha_t = self.alphas[t_idx]
            alpha_bar_t = self.alphas_cumprod[t_idx]
            beta_t = self.betas[t_idx]

            coef = beta_t / torch.sqrt(1 - alpha_bar_t)
            mean = (1 / torch.sqrt(alpha_t)) * (x - coef * eps_pred)
            if t_idx > 0:
                noise = torch.randn_like(x)
                x = mean + torch.sqrt(beta_t) * noise
            else:
                x = mean
        return x


def train(
    context_len: int = 168,
    horizon_multiplier: int = 3,
    n_timesteps: int = 200,
    epochs: int = 30,
    batch_size: int = 128,
    lr: float = 2e-4,
    csv_path: str = HOURLY_CSV,
    checkpoint_path: str = CHECKPOINT_PATH,
    log_ret_override: np.ndarray | None = None,
    bars_per_day_override: float | None = None,
) -> None:
    if log_ret_override is not None:
        log_ret, bars_per_day = log_ret_override, bars_per_day_override
    else:
        print(f"Loading hourly data from {csv_path} ...")
        log_ret, bars_per_day = load_hourly_log_returns(csv_path)
    horizon_len = int(round(bars_per_day * horizon_multiplier))
    print(f"bars/day ~= {bars_per_day:.1f}  ->  horizon_len (3 days) = {horizon_len} bars, context_len = {context_len} bars")

    mean, std = float(log_ret.mean()), float(log_ret.std() + 1e-12)
    log_ret_norm = (log_ret - mean) / std

    ctx_all, hor_all = make_windows(log_ret_norm, context_len, horizon_len)
    print(f"Training windows: {len(ctx_all)}")

    ctx_t = torch.tensor(ctx_all, device=DEVICE)
    hor_t = torch.tensor(hor_all, device=DEVICE)

    model = DenoiseNet(horizon_len=horizon_len, context_len=context_len).to(DEVICE)
    diffusion = GaussianDiffusion(n_timesteps, DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=lr)

    n = len(ctx_t)
    for epoch in range(1, epochs + 1):
        perm = torch.randperm(n, device=DEVICE)
        total_loss = 0.0
        n_batches = 0
        for start in range(0, n, batch_size):
            idx = perm[start : start + batch_size]
            x0 = hor_t[idx]
            ctx = ctx_t[idx]
            t = torch.randint(0, n_timesteps, (x0.shape[0],), device=DEVICE)
            noise = torch.randn_like(x0)
            x_noisy = diffusion.q_sample(x0, t, noise)

            eps_pred = model(x_noisy, t, ctx)
            loss = nn.functional.mse_loss(eps_pred, noise)

            opt.zero_grad()
            loss.backward()
            opt.step()

            total_loss += loss.item()
            n_batches += 1
        print(f"epoch {epoch:3d}/{epochs}  loss={total_loss / n_batches:.5f}")

    config = DiffusionConfig(
        context_len=context_len,
        horizon_len=horizon_len,
        bars_per_day=bars_per_day,
        n_timesteps=n_timesteps,
        mean=mean,
        std=std,
    )
    torch.save(
        {"model_state": model.state_dict(), "config": asdict(config)},
        checkpoint_path,
    )
    print(f"\nSaved checkpoint: {checkpoint_path}")


def load_model_and_config(checkpoint_path: str = CHECKPOINT_PATH) -> tuple[DenoiseNet, DiffusionConfig]:
    ckpt = torch.load(checkpoint_path, map_location=DEVICE)
    config = DiffusionConfig(**ckpt["config"])
    model = DenoiseNet(horizon_len=config.horizon_len, context_len=config.context_len).to(DEVICE)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model, config


def sample_scenarios(
    n_scenarios: int = 200,
    checkpoint_path: str = CHECKPOINT_PATH,
    recent_log_returns: np.ndarray | None = None,
) -> tuple[np.ndarray, DiffusionConfig]:
    """
    Returns an array of shape (n_scenarios, horizon_len) of *real-scale*
    (de-normalized) simulated hourly log returns, plus the config used
    (so callers know context_len / horizon_len / bars_per_day).
    """
    model, config = load_model_and_config(checkpoint_path)

    if recent_log_returns is None:
        log_ret, _ = load_hourly_log_returns()
        recent_log_returns = log_ret[-config.context_len :]

    if len(recent_log_returns) < config.context_len:
        raise ValueError(f"Need at least {config.context_len} recent hourly log returns for conditioning.")

    ctx_norm = (recent_log_returns[-config.context_len :] - config.mean) / config.std
    ctx_batch = torch.tensor(np.tile(ctx_norm, (n_scenarios, 1)), dtype=torch.float32, device=DEVICE)

    diffusion = GaussianDiffusion(config.n_timesteps, DEVICE)
    samples_norm = diffusion.p_sample_loop(model, ctx_batch, config.horizon_len)
    samples = samples_norm.cpu().numpy() * config.std + config.mean
    return samples, config


if __name__ == "__main__":
    if "--sample" in sys.argv:
        scenarios, cfg = sample_scenarios(n_scenarios=8)
        daily_chunks = np.array_split(scenarios, 3, axis=1)  # robust to horizon_len % 3 != 0
        daily_cum = np.stack([chunk.sum(axis=1) for chunk in daily_chunks], axis=1)
        print(f"config: {cfg}")
        print("Sample of 3-day cumulative log-return per scenario (first 8):")
        print(np.cumsum(daily_cum, axis=1))
    else:
        train()
