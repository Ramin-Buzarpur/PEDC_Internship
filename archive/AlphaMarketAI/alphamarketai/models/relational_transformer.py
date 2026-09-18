import torch
from torch import nn


class RelationalTransformer(nn.Module):
    def __init__(self, embed_dim: int, heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.attention = nn.MultiheadAttention(embed_dim, heads, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.feed_forward = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 2), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(embed_dim * 2, embed_dim),
        )
        self.norm2 = nn.LayerNorm(embed_dim)

    def forward(self, x: torch.Tensor, graph: torch.Tensor | None) -> torch.Tensor:
        mask = None
        if graph is not None:
            if graph.dim() == 4:
                graph = graph[0]
            adjacency = graph[..., 0].bool().to(x.device)
            mask = ~adjacency
        attended, _ = self.attention(x, x, x, attn_mask=mask, need_weights=False)
        x = self.norm1(x + attended)
        return self.norm2(x + self.feed_forward(x))
