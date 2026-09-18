import torch
from torch import nn
import torch.nn.functional as F

from .relational_transformer import RelationalTransformer


class TemporalEncoder(nn.Module):
    def __init__(self, features: int, hidden: int):
        super().__init__()
        self.network = nn.Sequential(
            nn.Conv1d(features, hidden, 3, padding=1), nn.GELU(),
            nn.Conv1d(hidden, hidden, 3, padding=1), nn.GELU(),
            nn.AdaptiveAvgPool1d(1),
        )

    def forward(self, x):
        return self.network(x).squeeze(-1)


class AlphaMaTCHS(nn.Module):
    """Compact MaTCHS-inspired deterministic relational forecaster."""
    def __init__(self, features=10, hidden=32, heads=4):
        super().__init__()
        self.encoder = TemporalEncoder(features, hidden)
        self.graph_encoder = RelationalTransformer(hidden, heads=heads)
        self.return_head = nn.Linear(hidden, 1)
        self.direction_head = nn.Linear(hidden, 1)
        self.mu_head = nn.Linear(hidden, 1)
        self.sigma_head = nn.Linear(hidden, 1)

    def forward(self, x, graph=None):
        batch, assets, features, steps = x.shape
        hidden = self.encoder(x.reshape(batch * assets, features, steps)).reshape(batch, assets, -1)
        hidden = self.graph_encoder(hidden, graph)
        return {
            "return": self.return_head(hidden).squeeze(-1),
            "direction": self.direction_head(hidden).squeeze(-1),
            "mu": self.mu_head(hidden).squeeze(-1),
            "sigma": F.softplus(self.sigma_head(hidden)).squeeze(-1) + 1e-4,
            "embedding": hidden,
        }
