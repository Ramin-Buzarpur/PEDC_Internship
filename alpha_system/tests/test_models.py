from __future__ import annotations

import torch

from src.models.matchs import MaTCHS, MaskedRelationalAttention
from src.models.temporal import SinusoidalTimeEmbedding


def test_sinusoidal_embedding_range():
    emb = SinusoidalTimeEmbedding(16)
    t = torch.tensor([0.0, 5.0, 59.0])
    out = emb(t)
    assert out.shape == (3, 16)
    assert (out.abs() <= 1.0 + 1e-6).all()


def test_matchs_forward_shape():
    B, C, N, T = 2, 6, 8, 40
    model = MaTCHS(num_channels=C, seq_len_total=T, d_channels=32, mrt_layers=1, heads_masked=2, heads_free=4, num_groups=2)
    x = torch.randn(B, C, N, T)
    cond = x.clone()
    cond[..., -10:] = 0
    t = torch.randint(0, 60, (B,))
    rel = torch.ones(N, N, 2) * 0.5
    out = model(x, t, cond, rel)
    assert out.shape == x.shape
    assert torch.isfinite(out).all()


def test_masked_attention_blocks_disconnected_nodes():
    torch.manual_seed(0)
    attn = MaskedRelationalAttention(embed_dim=32, heads_masked=2, heads_free=0, num_groups=2)
    N, D = 5, 32
    x = torch.randn(1, N, D)
    rel = torch.zeros(N, N, 2)
    rel[0, 0] = rel[4, 4] = 1
    out_isolated = attn(x, rel)
    assert torch.isfinite(out_isolated).all()
    rel_full = torch.ones(N, N, 2)
    out_full = attn(x, rel_full)
    assert not torch.allclose(out_isolated, out_full)


def test_gradient_flows_through_model():
    model = MaTCHS(num_channels=5, seq_len_total=40, d_channels=24, mrt_layers=1, heads_masked=2, heads_free=2, num_groups=2)
    x = torch.randn(2, 5, 4, 40)
    cond = x.clone()
    t = torch.randint(0, 50, (2,))
    rel = torch.ones(4, 4, 2)
    loss = ((model(x, t, cond, rel) - x) ** 2).mean()
    loss.backward()
    grads_ok = any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.parameters())
    assert grads_ok
