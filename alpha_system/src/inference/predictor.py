from __future__ import annotations

import numpy as np
import torch

from ..data.dataset import build_channels
from ..models.diffusion import ConditionalDiffusion


class ScenarioPredictor:
    def __init__(self, diffusion: ConditionalDiffusion, arrays: dict[str, np.ndarray], sentiment: np.ndarray | None, seq_len_past: int, seq_len_future: int, device: torch.device):
        self.diffusion = diffusion
        full = build_channels(arrays, sentiment)
        model_channels = getattr(diffusion.model, "C", full.shape[-1])
        self.data = full[..., :model_channels]
        self.L = seq_len_past
        self.H = seq_len_future
        self.device = device

    def _condition(self, t_end: int) -> torch.Tensor:
        window = self.data[t_end - self.L + 1 : t_end + 1]
        mean = window.mean(axis=(0, 1), keepdims=True)
        std = window.std(axis=(0, 1), keepdims=True) + 1e-5
        norm = (window - mean) / std
        cond = np.zeros((self.L + self.H, norm.shape[1], norm.shape[2]), dtype=np.float32)
        cond[: self.L] = norm[: self.L]
        tensor = torch.from_numpy(cond.transpose(2, 1, 0).copy()).unsqueeze(0).to(self.device)
        return tensor, mean.astype(np.float32), std.astype(np.float32)

    @torch.no_grad()
    def predict(self, t_end: int, num_samples: int = 16, ddim_steps: int = 15, rel: torch.Tensor | None = None) -> dict:
        cond_np, mean, std = self._condition(t_end)
        B, C, N, T = cond_np.shape
        conds = cond_np.repeat(num_samples, 1, 1, 1)
        rel_b = rel if rel is not None else torch.ones(N, N, 1, device=self.device)
        x = self.diffusion.ddim_sample(conds, rel_b, steps=ddim_steps)
        fut = x[..., -self.H :]
        m_t = torch.as_tensor(mean.reshape(1, C, 1, 1), device=self.device)
        s_t = torch.as_tensor(std.reshape(1, C, 1, 1), device=self.device)
        fut_denorm = fut * s_t + m_t
        next_day_ret = fut_denorm[:, 3, :, 0].cpu().numpy()
        horizon_cum = np.prod(1.0 + fut_denorm[:, 3, :, :].cpu().numpy(), axis=0) - 1.0
        vol_pred = fut_denorm[:, 3, :, :].cpu().numpy().std(axis=0).mean(axis=-1)
        return {
            "samples": fut_denorm.cpu().numpy(),
            "next_day_return_mean": next_day_ret.mean(axis=0),
            "next_day_return_std": next_day_ret.std(axis=0),
            "horizon_cum_return_mean": horizon_cum.mean(axis=0),
            "horizon_cum_return_std": horizon_cum.std(axis=0),
            "path_vol": vol_pred,
        }
