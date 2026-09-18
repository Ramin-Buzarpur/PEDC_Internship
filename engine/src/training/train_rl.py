from __future__ import annotations

import os

import numpy as np
import torch

from ..backtest.metrics import annualized_sharpe
from ..data.dataset import build_splits
from ..models.rl_agent import PPOAgent, PortfolioEnv, train_ppo


STATE_FEATURES = ["mom_5", "mom_21", "macd_signal", "ret_1d", "vol_z", "body"]


def build_state_feats(feats: dict[str, np.ndarray], sentiment: np.ndarray | None, target_channels: int | None = None) -> np.ndarray:
    parts = [feats[name] for name in STATE_FEATURES if name in feats]
    if sentiment is not None and sentiment.shape == feats["ret_1d"].shape:
        parts.append(sentiment)
    arr = np.stack(parts, axis=-1)
    if target_channels is not None and arr.shape[2] < target_channels:
        pad = target_channels - arr.shape[2]
        arr = np.concatenate([arr, np.zeros(arr.shape[:2] + (pad,), dtype=arr.dtype)], axis=-1)
    return arr


def make_env(cfg, splits, returns_daily, state_feats, daily_vol, start, end, base_signal=None, random_start=False):
    rcfg = cfg.section("rl")
    return PortfolioEnv(
        returns=returns_daily,
        state_feats=state_feats,
        daily_vol=daily_vol,
        start=start,
        end=end,
        vol_target_annual=float(rcfg.get("vol_target_annual", 0.15)),
        max_leverage=float(rcfg.get("max_leverage", 1.0)),
        max_weight=float(rcfg.get("max_weight", 0.25)),
        cost_bps=float(rcfg.get("cost_bps", 5.0)),
        drawdown_penalty=float(rcfg.get("drawdown_penalty", 2.0)),
        random_start=random_start,
        base_signal=base_signal,
        resid_scale=float(rcfg.get("resid_scale", 1.0)),
    )


@torch.no_grad()
def evaluate_deterministic(agent: PPOAgent, env: PortfolioEnv) -> tuple[float, float]:
    s = env.reset()
    rets = []
    while True:
        a, _, _ = agent.act(s, deterministic=True)
        s, r, done, info = env.step(np.asarray(a, dtype=np.float32))
        rets.append(info["port_ret"])
        if done or env.t >= env.end - 1:
            break
    rets = np.asarray(rets)
    return annualized_sharpe(rets), float(env.equity)


def train_rl_agent(cfg, feats: dict[str, np.ndarray], arrays: dict[str, np.ndarray], sentiment: np.ndarray | None, device: torch.device, log=print):
    rcfg = cfg.section("rl")
    tcfg = cfg.section("training")
    scfg = cfg.section("dataset")

    L = int(scfg.get("seq_len_past", 30))
    T_days = arrays["close_adj"].shape[0]
    splits = build_splits(T_days, float(tcfg.get("val_ratio", 0.15)), 0.15, int(tcfg.get("purge_gap", 40)))

    returns_daily = (arrays["close_adj"][1:] / arrays["close_adj"][:-1] - 1.0).astype(np.float32)
    returns_daily = np.vstack([np.zeros((1, returns_daily.shape[1]), dtype=np.float32), returns_daily])

    from ..backtest.metrics import zscore
    from ..data.features import ewma_vol

    sigma_ann = ewma_vol(arrays["close_adj"], span=32)
    daily_vol = sigma_ann / np.sqrt(252.0)

    state_feats = build_state_feats(feats, sentiment)
    base_signal = np.clip(zscore(feats["mom_21"]), -3.0, 3.0) / 3.0

    train_lo = splits["train"][0] + L
    env = make_env(cfg, splits, returns_daily, state_feats, daily_vol, train_lo, splits["val"][0], base_signal=base_signal, random_start=True)
    agent = PPOAgent(env.state_dim, env.n_assets, rcfg, device=device)
    torch.nn.init.zeros_(agent.net.mu_head.weight)
    torch.nn.init.zeros_(agent.net.mu_head.bias)
    log(f"PPO state dim: {env.state_dim}, actions: {env.n_assets} (residual policy over momentum)")

    sr0, eq0 = evaluate_deterministic(agent, make_env(cfg, splits, returns_daily, state_feats, daily_vol, splits["val"][0] + L, splits["test"][0], base_signal=base_signal))
    log(f"momentum-only baseline on val: Sharpe {sr0:.2f} | equity {eq0:.4f}")

    out_dir = cfg.get("paths.outputs", "./outputs")
    model_dir = os.path.join(out_dir, "models")
    os.makedirs(model_dir, exist_ok=True)
    path = os.path.join(model_dir, "ppo_actor.pt")

    import time

    total_updates = int(rcfg.get("total_updates", 120))
    eval_every = max(5, total_updates // 12)
    best_sharpe = -np.inf
    t0 = time.time()
    history = []
    for up in range(total_updates):
        from ..models.rl_agent import collect_rollout

        states, actions, adv, rets_, logps, infos = collect_rollout(env, agent, agent.steps_per_update)
        stats = agent.update(states, actions, adv, rets_, logps)
        eqs = [i["equity"] for i in infos] or [env.equity]
        history.append(float(np.mean(eqs)))
        if (up + 1) % eval_every == 0:
            sr, eq = evaluate_deterministic(agent, make_env(cfg, splits, returns_daily, state_feats, daily_vol, splits["val"][0] + L, splits["test"][0], base_signal=base_signal))
            marker = ""
            if sr > best_sharpe:
                best_sharpe = sr
                agent.save(path)
                marker = " *saved*"
            log(f"PPO update {up + 1}/{total_updates} | train-eq {history[-1]:.3f} | val SR {sr:.2f} eq {eq:.3f} | H {stats['entropy']:.2f}{marker}")
    if best_sharpe == -np.inf:
        agent.save(path)
    log(f"PPO finished in {time.time() - t0:.0f}s | best val Sharpe {best_sharpe:.2f} | saved {path}")

    return {"agent": agent, "env": env, "splits": splits, "daily_vol": daily_vol, "returns_daily": returns_daily, "history": history, "best_val_sharpe": best_sharpe}
