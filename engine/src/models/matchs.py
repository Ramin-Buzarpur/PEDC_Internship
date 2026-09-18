from __future__ import annotations

import math

import torch
import torch.nn as nn

from ..models.temporal import SinusoidalTimeEmbedding, TemporalEncoder


class MaskedRelationalAttention(nn.Module):
    def __init__(self, embed_dim: int, heads_masked: int, heads_free: int, num_groups: int, dropout: float = 0.1):
        super().__init__()
        assert heads_masked <= num_groups or heads_masked == 0
        self.heads_masked = heads_masked
        self.heads_free = heads_free
        self.H = heads_masked + heads_free
        self.embed_dim = embed_dim
        self.head_dim = embed_dim // self.H
        assert self.head_dim * self.H == embed_dim, "embed_dim must be divisible by total heads"
        self.qkv = nn.Linear(embed_dim, 3 * embed_dim)
        self.proj = nn.Linear(embed_dim, embed_dim)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, rel: torch.Tensor | None) -> torch.Tensor:
        B, N, D = x.shape
        Hd, dh = self.H, self.head_dim
        qkv = self.qkv(x).reshape(B, N, 3, Hd, dh).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        scores = (q @ k.transpose(-2, -1)) / math.sqrt(dh)
        if rel is not None and self.heads_masked > 0:
            G = min(rel.shape[-1], self.heads_masked)
            mask = torch.zeros(Hd, N, N, device=x.device, dtype=scores.dtype)
            fill = torch.finfo(scores.dtype).min
            for g in range(G):
                blocked = rel[:, :, g] < 0.5
                mask[g].masked_fill_(blocked, fill)
            mask[self.heads_masked :] = 0.0
            scores = scores + mask.unsqueeze(0)
        attn = scores.softmax(dim=-1)
        attn = self.drop(attn)
        out = (attn @ v).transpose(1, 2).reshape(B, N, D)
        return self.proj(out)


class MRTBlock(nn.Module):
    def __init__(self, embed_dim: int, heads_masked: int, heads_free: int, num_groups: int, dropout: float = 0.1):
        super().__init__()
        self.attn = MaskedRelationalAttention(embed_dim, heads_masked, heads_free, num_groups, dropout)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.ffn = nn.Sequential(nn.Linear(embed_dim, embed_dim * 4), nn.GELU(), nn.Dropout(dropout), nn.Linear(embed_dim * 4, embed_dim))
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, rel: torch.Tensor | None) -> torch.Tensor:
        h = self.norm1(x + self.drop(self.attn(x, rel)))
        return self.norm2(h + self.drop(self.ffn(h)))


class MaTCHS(nn.Module):
    HEAD_DIM = 64

    def __init__(
        self,
        num_channels: int,
        seq_len_total: int,
        d_channels: int = 48,
        mrt_layers: int = 2,
        heads_masked: int = 2,
        heads_free: int = 4,
        num_groups: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.C = num_channels
        self.T = seq_len_total
        self.d = d_channels
        total_heads = heads_masked + heads_free
        self.mrt_dim = self.HEAD_DIM * total_heads
        self.time_mlp = nn.Sequential(
            SinusoidalTimeEmbedding(d_channels),
            nn.Linear(d_channels, d_channels * 4),
            nn.GELU(),
            nn.Linear(d_channels * 4, d_channels),
        )
        self.enc_in = TemporalEncoder(2 * num_channels, d_channels, dropout)
        flat_dim = d_channels * seq_len_total
        self.in_proj = nn.Linear(flat_dim, self.mrt_dim)
        self.mrt = nn.ModuleList(
            [
                MRTBlock(self.mrt_dim, heads_masked, heads_free, num_groups, dropout)
                for _ in range(mrt_layers)
            ]
        )
        self.out_proj = nn.Linear(self.mrt_dim, flat_dim)
        self.dec = TemporalEncoder(d_channels, d_channels, dropout)
        self.head = nn.Conv1d(d_channels, num_channels, kernel_size=1)

    def forward(self, noisy_x: torch.Tensor, t: torch.Tensor, condition: torch.Tensor, rel: torch.Tensor) -> torch.Tensor:
        B = noisy_x.shape[0]
        N = noisy_x.shape[2]
        xin = torch.cat([noisy_x, condition], dim=1)
        x = xin.permute(0, 2, 1, 3).reshape(B * N, -1, self.T)
        h = self.enc_in(x)
        t_emb = self.time_mlp(t).repeat_interleave(N, dim=0).unsqueeze(-1)
        h = h + t_emb
        flat = h.reshape(B, N, -1)
        flat = self.in_proj(flat)
        for blk in self.mrt:
            flat = blk(flat, rel)
        flat = self.out_proj(flat)
        h = flat.reshape(B * N, self.d, self.T)
        h = self.dec(h)
        out = self.head(h).reshape(B, N, self.C, self.T)
        return out.permute(0, 2, 1, 3)
