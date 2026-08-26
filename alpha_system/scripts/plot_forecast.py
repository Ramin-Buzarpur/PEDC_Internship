from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt

from src.config import load_config
from src.inference.predictor import ScenarioPredictor
from src.models.diffusion import ConditionalDiffusion
from src.models.matchs import MaTCHS
from src.pipeline import get_device, nan_guard, prepare_data
from src.utils.runtime import set_seed


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--num-stocks", type=int, default=4)
    args = ap.parse_args()

    cfg = load_config(args.config)
    set_seed(int(cfg.get("project.seed", 42)))
    out_dir = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "outputs"))
    device = get_device(cfg)

    dates, symbols, arrays, sentiment = prepare_data(cfg, fetch_news=False)
    arrays = nan_guard(arrays)

    scfg = cfg.section("dataset")
    dcfg = cfg.section("diffusion")
    L, H = int(scfg.get("seq_len_past", 30)), int(scfg.get("seq_len_future", 10))
    model_path = os.path.join(out_dir, "models", "matchs_diffusion.pt")
    ckpt = torch.load(model_path, map_location=device)
    C = int(ckpt.get("num_channels", 6))
    T_total = int(ckpt.get("seq_len_total", L + H))
    model = MaTCHS(num_channels=C, seq_len_total=T_total, d_channels=int(cfg.section("model").get("d_channels", 48))).to(device)
    model.load_state_dict(ckpt["state_dict"])
    diffusion = ConditionalDiffusion(model, num_timesteps=int(ckpt.get("num_timesteps", 60)), seq_len_future=H, device=device)

    predictor = ScenarioPredictor(diffusion, arrays, sentiment, L, H, device)
    t_last = arrays["close_adj"].shape[0] - 1
    scen = predictor.predict(t_last, num_samples=int(dcfg.get("num_samples", 32)), ddim_steps=int(dcfg.get("ddim_steps", 15)))

    vol_rank = np.argsort(-np.nanstd(arrays["close_adj"][t_last - 60 : t_last] / arrays["close_adj"][t_last - 61 : t_last - 1] - 1.0, axis=0))
    picks = list(vol_rank[: args.num_stocks])

    past_close = arrays["close_adj"][t_last - L + 1 : t_last + 1]
    true_fut = arrays["close_adj"][t_last + 1 : t_last + 1 + H] if t_last + 1 + H <= len(dates) else None
    samples = scen["samples"][:, 3, :, :]
    c_prev = arrays["close_adj"][t_last]

    fig, axes = plt.subplots(min(args.num_stocks, 4), 1, figsize=(12, 3.2 * min(args.num_stocks, 4)), sharex=False)
    if args.num_stocks == 1:
        axes = [axes]
    x_past = np.arange(L)
    x_fut = np.arange(L, L + H)
    for ax, si in zip(np.atleast_1d(axes), picks):
        p = past_close[:, si]
        ax.plot(x_past, p, color="#1f77b4", lw=2, label="Past (known)")
        rets_s = scen["samples"][:, 3, si, :]
        paths = c_prev[si] * np.cumprod(1.0 + rets_s, axis=-1)
        for s in range(paths.shape[0]):
            ax.plot(x_fut, paths[s], color="#d62728", alpha=0.10, lw=1)
        lo, hi = np.percentile(paths, [10, 90], axis=0)
        med = np.median(paths, axis=0)
        ax.plot(x_fut, med, color="#d62728", lw=2.5, ls="--", label="Median forecast")
        ax.fill_between(x_fut, lo, hi, color="#d62728", alpha=0.18, label="80% band")
        if true_fut is not None:
            truth = true_fut[:, si]
            ax.plot(x_fut, truth, color="black", lw=2, marker="o", ms=3, label="Realized")
        ax.axvline(L - 1, color="gray", ls=":", lw=1)
        ax.set_title(f"{symbols[si]} — {scen['path_vol'][si]:.3f} path-vol")
        ax.legend(fontsize=8, loc="best")
        ax.grid(alpha=0.3)
    axes[-1].xaxis.set_major_formatter(plt.NullFormatter())
    plt.suptitle(f"PEDC Alpha probabilistic forecasts ({dates[t_last].date()} close)", fontsize=13)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    out_path = os.path.join(out_dir, "plots", "forecast_fan.png")
    plt.savefig(out_path, dpi=150)
    print(f"saved {out_path}")


if __name__ == "__main__":
    main()
