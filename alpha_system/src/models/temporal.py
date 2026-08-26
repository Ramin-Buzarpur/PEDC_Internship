from __future__ import annotations

import math

import torch
import torch.nn as nn


class SinusoidalTimeEmbedding(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        half = self.dim // 2
        freqs = torch.exp(-math.log(10000.0) * torch.arange(half, device=t.device, dtype=torch.float32) / max(half - 1, 1))
        args = t.float()[:, None] * freqs[None, :]
        return torch.cat([args.sin(), args.cos()], dim=-1)


class AttDiCEm(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, dilation: int = 1, kernel: int = 3):
        super().__init__()
        self.pad = (kernel - 1) * dilation
        self.conv = nn.Conv1d(in_ch, out_ch, kernel, dilation=dilation)
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = nn.functional.pad(x, (self.pad, 0))
        return self.act(self.conv(x))


class TemporalEncoder(nn.Module):
    def __init__(self, in_ch: int, d: int, dropout: float = 0.1):
        super().__init__()
        self.blocks = nn.ModuleList(
            [
                AttDiCEm(in_ch, d, dilation=1),
                ResidualBlock(d, dilation=2, dropout=dropout),
                ResidualBlock(d, dilation=4, dropout=dropout),
            ]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = x
        for b in self.blocks:
            h = b(h)
        return h


class ResidualBlock(nn.Module):
    def __init__(self, d: int, dilation: int, dropout: float = 0.1):
        super().__init__()
        self.conv1 = AttDiCEm(d, d, dilation=dilation)
        self.conv2 = AttDiCEm(d, d, dilation=1)
        self.norm = nn.GroupNorm(1, d)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.conv2(self.drop(self.conv1(x)))
        return self.norm(x + h)
