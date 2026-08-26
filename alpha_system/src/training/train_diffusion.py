from __future__ import annotations

import os
import time

import numpy as np
import torch
from torch.utils.data import DataLoader

from ..data.dataset import StockDiffusionDataset, build_splits
from ..data.graph import build_relation_matrix
from ..models.diffusion import ConditionalDiffusion
from ..models.matchs import MaTCHS


def train_diffusion_model(cfg, arrays: dict[str, np.ndarray], sentiment: np.ndarray | None, device: torch.device, log=print):
    tcfg = cfg.section("training")
    dcfg = cfg.section("diffusion")
    mcfg = cfg.section("model")
    scfg = cfg.section("dataset")
    gcfg = cfg.section("graph")

    L = int(scfg.get("seq_len_past", 30))
    H = int(scfg.get("seq_len_future", 10))
    T_days = arrays["close_adj"].shape[0]
    splits = build_splits(T_days, float(tcfg.get("val_ratio", 0.15)), 0.15, int(tcfg.get("purge_gap", 40)))
    log(f"splits: train {splits['train']}, val {splits['val']}, test {splits['test']}")

    train_ds = StockDiffusionDataset(arrays, sentiment, *splits["train"], L, H)
    val_ds = StockDiffusionDataset(
        arrays, sentiment, *splits["val"], L, H,
        precomputed_weights=np.ones(max(1, splits["val"][1] - splits["val"][0] - L - H + 1), dtype=np.float32),
    )
    C = train_ds.data.shape[-1]
    log(f"channels per stock: {C}")

    rel_np = build_relation_matrix(arrays["close_adj"], 0, splits["train"][1] + L + H, int(gcfg.get("corr_window", 250)), float(gcfg.get("corr_threshold", 0.5)))
    rel = torch.from_numpy(rel_np).to(device)

    model = MaTCHS(
        num_channels=C,
        seq_len_total=L + H,
        d_channels=int(mcfg.get("d_channels", 48)),
        mrt_layers=int(mcfg.get("mrt_layers", 2)),
        heads_masked=min(int(mcfg.get("heads_masked", 2)), rel_np.shape[-1]),
        heads_free=int(mcfg.get("heads_free", 4)),
        num_groups=rel_np.shape[-1],
        dropout=float(mcfg.get("dropout", 0.1)),
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    log(f"MaTCHS parameters: {n_params:,}")

    diffusion = ConditionalDiffusion(
        model,
        num_timesteps=int(dcfg.get("num_timesteps", 60)),
        beta_start=float(dcfg.get("beta_start", 1e-4)),
        beta_end=float(dcfg.get("beta_end", 0.2)),
        seq_len_future=H,
        device=device,
    )

    bs = int(tcfg.get("batch_size", 16))
    train_dl = DataLoader(train_ds, batch_size=bs, shuffle=True, drop_last=True, num_workers=0)
    val_dl = DataLoader(val_ds, batch_size=bs, shuffle=False, num_workers=0)

    opt = torch.optim.AdamW(model.parameters(), lr=float(tcfg.get("lr", 5e-4)), weight_decay=float(tcfg.get("weight_decay", 1e-4)))
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=int(tcfg.get("epochs_diffusion", 25)))

    best_val = np.inf
    best_state = None
    patience = int(tcfg.get("patience", 6))
    bad = 0
    t0 = time.time()
    for epoch in range(int(tcfg.get("epochs_diffusion", 25))):
        model.train()
        tr_loss, nb = 0.0, 0
        for x_all, w in train_dl:
            x_all = x_all.to(device, non_blocking=True)
            w = w.to(device)
            cond = x_all.clone()
            cond[..., -H:] = 0.0
            loss = diffusion.loss(x_all, cond, rel, sample_weight=w)
            if not torch.isfinite(loss):
                continue
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(tcfg.get("grad_clip", 1.0)))
            opt.step()
            tr_loss += float(loss.item())
            nb += 1
        sched.step()

        model.eval()
        va_loss, vnb = 0.0, 0
        with torch.no_grad():
            for x_all, _ in val_dl:
                x_all = x_all.to(device)
                cond = x_all.clone()
                cond[..., -H:] = 0.0
                l = diffusion.loss(x_all, cond, rel)
                if torch.isfinite(l):
                    va_loss += float(l.item())
                    vnb += 1
        tr_loss = tr_loss / max(nb, 1)
        va_loss = va_loss / max(vnb, 1)
        log(f"[diffusion] epoch {epoch + 1:03d} | train {tr_loss:.5f} | val {va_loss:.5f} | {time.time() - t0:.0f}s")
        if va_loss < best_val:
            best_val = va_loss
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            bad = 0
        else:
            bad += 1
            if bad >= patience:
                log("early stopping")
                break

    out_dir = cfg.get("paths.outputs", "./outputs")
    model_dir = os.path.join(out_dir, "models")
    os.makedirs(model_dir, exist_ok=True)
    path = os.path.join(model_dir, "matchs_diffusion.pt")
    if best_state is not None:
        model.load_state_dict(best_state)
    torch.save(
        {
            "state_dict": model.state_dict(),
            "num_channels": C,
            "seq_len_total": L + H,
            "seq_len_future": H,
            "num_timesteps": int(dcfg.get("num_timesteps", 60)),
        },
        path,
    )
    log(f"saved diffusion model to {path} (best val loss {best_val:.5f})")

    return {"diffusion": diffusion, "splits": splits, "rel_matrix": rel, "best_val_loss": best_val}
