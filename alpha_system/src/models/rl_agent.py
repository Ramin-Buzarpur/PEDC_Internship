from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Normal


class PortfolioEnv:
    def __init__(
        self,
        returns: np.ndarray,
        state_feats: np.ndarray,
        daily_vol: np.ndarray,
        start: int,
        end: int,
        vol_target_annual: float = 0.15,
        max_leverage: float = 1.0,
        max_weight: float = 0.25,
        cost_bps: float = 5.0,
        drawdown_penalty: float = 2.0,
        random_start: bool = False,
        min_episode_len: int = 400,
        base_signal: np.ndarray | None = None,
        resid_scale: float = 1.0,
    ):
        self.rets = returns.astype(np.float32)
        self.feats = state_feats.astype(np.float32)
        self.vol = np.maximum(daily_vol.astype(np.float32), 1e-5)
        self.base_signal = None if base_signal is None else base_signal.astype(np.float32)
        self.resid_scale = float(resid_scale)
        self.base_start, self.end = start, end
        self.random_start = random_start
        self.min_episode_len = int(min_episode_len)
        self.rng = np.random.default_rng(123)
        self.vol_target_daily = float(vol_target_annual / np.sqrt(252.0))
        self.max_leverage = float(max_leverage)
        self.max_weight = float(max_weight)
        self.cost = float(cost_bps) / 1e4
        self.dd_pen = float(drawdown_penalty)
        self.n_assets = self.rets.shape[1]
        self.reset()

    def reset(self) -> np.ndarray:
        if self.random_start:
            hi = max(self.base_start + 1, self.end - self.min_episode_len)
            self.t = int(self.rng.integers(self.base_start, hi + 1))
        else:
            self.t = self.base_start
        self.weights = np.zeros(self.n_assets, dtype=np.float32)
        self.equity = 1.0
        self.peak = 1.0
        self.drawdown = 0.0
        return self._state()

    def _state(self) -> np.ndarray:
        f = self.feats[self.t]
        vol_n = (self.vol[self.t] / 0.02).astype(np.float32)
        per_asset = np.concatenate([f, vol_n[:, None], self.weights[:, None]], axis=1)
        glob = np.array([self.weights.sum(), self.drawdown], dtype=np.float32)
        return np.concatenate([per_asset.reshape(-1), glob])

    @property
    def state_dim(self) -> int:
        return self.n_assets * (self.feats.shape[2] + 2) + 2

    def _target_weights(self, action: np.ndarray) -> np.ndarray:
        a = np.tanh(np.asarray(action, dtype=np.float32))
        if self.base_signal is not None:
            sig = self.base_signal[self.t] + self.resid_scale * a
        else:
            sig = a
        e = sig * (self.vol_target_daily / self.vol[self.t])
        e = np.clip(e, -self.max_weight, self.max_weight)
        gross = np.abs(e).sum()
        if gross > self.max_leverage:
            e *= self.max_leverage / (gross + 1e-9)
        return e.astype(np.float32)

    def step(self, action: np.ndarray):
        if self.t >= self.end - 1:
            return self._state(), 0.0, True, {}
        e = self._target_weights(np.asarray(action, dtype=np.float32))
        turnover = np.abs(e - self.weights).sum()
        cost = turnover * self.cost
        port_ret = float((e * self.rets[self.t + 1]).sum() - cost)
        self.equity *= float(np.exp(np.log1p(np.clip(port_ret, -0.9, 10.0))))
        self.peak = max(self.peak, self.equity)
        new_dd = 1.0 - self.equity / self.peak
        reward = float(np.log1p(np.clip(port_ret, -0.9, 10.0)) - self.dd_pen * max(0.0, new_dd - self.drawdown))
        self.drawdown = new_dd
        self.weights = e
        self.t += 1
        done = self.t >= self.end - 1
        info = {"port_ret": port_ret, "equity": self.equity}
        return self._state(), reward, done, info


class ActorCritic(nn.Module):
    def __init__(self, state_dim: int, n_actions: int, hidden: int = 128, layers: int = 2):
        super().__init__()
        mods: list[nn.Module] = []
        d_in = state_dim
        for _ in range(layers):
            mods += [nn.Linear(d_in, hidden), nn.LayerNorm(hidden), nn.Tanh()]
            d_in = hidden
        self.body = nn.Sequential(*mods)
        self.n_actions = n_actions
        self.mu_head = nn.Linear(hidden, n_actions)
        self.v_head = nn.Linear(hidden, 1)
        self.log_std = nn.Parameter(torch.full((n_actions,), -0.5))

    def forward(self, x: torch.Tensor):
        h = self.body(x)
        mu = torch.tanh(self.mu_head(h))
        v = self.v_head(h).squeeze(-1)
        return mu, v

    def dist(self, x: torch.Tensor) -> Normal:
        mu, _ = self.forward(x)
        std = torch.exp(self.log_std).clamp(0.05, 2.0)
        return Normal(mu, std.expand_as(mu))


def gae(rewards, values, dones, last_value, gamma: float, lam: float):
    T = len(rewards)
    adv = np.zeros(T, dtype=np.float32)
    last = 0.0
    for i in reversed(range(T)):
        next_v = last_value if i == T - 1 else values[i + 1]
        nonterm = 0.0 if dones[i] else 1.0
        delta = rewards[i] + gamma * next_v * nonterm - values[i]
        last = adv[i] = delta + gamma * lam * nonterm * last
    return adv


class PPOAgent:
    def __init__(self, state_dim: int, n_actions: int, cfg: dict, device: torch.device | str = "cpu"):
        hidden = int(cfg.get("hidden_dim", 128))
        layers = int(cfg.get("layers", 2))
        self.net = ActorCritic(state_dim, n_actions, hidden, layers).to(device)
        self.opt = torch.optim.Adam(self.net.parameters(), lr=float(cfg.get("lr", 3e-4)))
        self.gamma = float(cfg.get("gamma", 0.99))
        self.lam = float(cfg.get("gae_lambda", 0.95))
        self.clip = float(cfg.get("clip_eps", 0.2))
        self.epochs = int(cfg.get("epochs_per_update", 4))
        self.entropy_coef = float(cfg.get("entropy_coef", 0.01))
        self.value_coef = float(cfg.get("value_coef", 0.5))
        self.steps_per_update = int(cfg.get("steps_per_update", 512))
        self.n_actions = n_actions
        self.state_dim = state_dim
        self.device = torch.device(device)
        with torch.no_grad():
            self.net.log_std.fill_(-1.0)

    @torch.no_grad()
    def act(self, state: np.ndarray, deterministic: bool = False):
        s = torch.as_tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
        dist = self.net.dist(s)
        _, v = self.net(s)
        if deterministic:
            a = dist.mean
        else:
            a = dist.sample()
        logp = dist.log_prob(a).sum(-1)
        return a.squeeze(0).cpu().numpy(), float(v.item()), float(logp.item())

    def update(self, states, actions, advantages, returns, old_logp, batch_size: int = 256):
        n = len(states)
        idx_all = np.arange(n)
        stats = {"pi_loss": 0.0, "v_loss": 0.0, "entropy": 0.0}
        st = torch.as_tensor(np.asarray(states), dtype=torch.float32, device=self.device)
        ac = torch.as_tensor(np.asarray(actions), dtype=torch.float32, device=self.device)
        ad = torch.as_tensor(np.asarray(advantages), dtype=torch.float32, device=self.device)
        rt = torch.as_tensor(np.asarray(returns), dtype=torch.float32, device=self.device)
        ol = torch.as_tensor(np.asarray(old_logp), dtype=torch.float32, device=self.device)
        ad = (ad - ad.mean()) / (ad.std() + 1e-8)
        for _ in range(self.epochs):
            np.random.shuffle(idx_all)
            for b in range(0, n, batch_size):
                bi = idx_all[b : b + batch_size]
                dist = self.net.dist(st[bi])
                logp = dist.log_prob(ac[bi]).sum(-1)
                ratio = (logp - ol[bi]).exp()
                s1 = ratio * ad[bi]
                s2 = torch.clamp(ratio, 1 - self.clip, 1 + self.clip) * ad[bi]
                pi_loss = -torch.min(s1, s2).mean()
                _, v = self.net(st[bi])
                v_loss = 0.5 * ((v - rt[bi]) ** 2).mean()
                ent = dist.entropy().sum(-1).mean()
                loss = pi_loss + self.value_coef * v_loss - self.entropy_coef * ent
                self.opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.net.parameters(), 1.0)
                self.opt.step()
                stats["pi_loss"] += float(pi_loss.item())
                stats["v_loss"] += float(v_loss.item())
                stats["entropy"] += float(ent.item())
        k = max(1, self.epochs * max(1, n // batch_size))
        return {key: val / k for key, val in stats.items()}

    def save(self, path: str):
        torch.save(
            {
                "state_dict": self.net.state_dict(),
                "state_dim": self.state_dim,
                "n_actions": self.n_actions,
            },
            path,
        )

    @staticmethod
    def dims_from_path(path: str) -> tuple[int, int]:
        obj = torch.load(path, map_location="cpu")
        if isinstance(obj, dict) and "state_dict" in obj:
            return int(obj["state_dim"]), int(obj["n_actions"])
        first = obj["body.0.weight"]
        return int(first.shape[1]), int(obj["mu_head.weight"].shape[0])

    def load(self, path: str):
        obj = torch.load(path, map_location=self.device)
        sd = obj["state_dict"] if isinstance(obj, dict) and "state_dict" in obj else obj
        self.net.load_state_dict(sd)
        self.net.eval()


def collect_rollout(env: PortfolioEnv, agent: PPOAgent, steps: int):
    states, actions, logps, rewards, dones, values = [], [], [], [], [], []
    state = env.reset()
    ep_infos = []
    for _ in range(steps):
        a, v, lp = agent.act(state)
        next_state, r, done, info = env.step(np.asarray(a, dtype=np.float32))
        states.append(state)
        actions.append(np.asarray(a, dtype=np.float32))
        logps.append(lp)
        rewards.append(r)
        dones.append(done)
        values.append(v)
        state = next_state
        if done or env.t >= env.end - 1:
            ep_infos.append({"equity": info.get("equity", env.equity)})
            state = env.reset()
    last_v = values[-1] if not dones[-1] else 0.0
    adv = gae(rewards, values, dones, last_v, agent.gamma, agent.lam)
    rets = (adv + np.asarray(values)).astype(np.float32)
    return states, actions, adv, rets, logps, ep_infos


def train_ppo(env: PortfolioEnv, agent: PPOAgent, total_updates: int, log=print):
    import time

    best_equity = -np.inf
    history = []
    t0 = time.time()
    for up in range(total_updates):
        states, actions, adv, rets, logps, infos = collect_rollout(env, agent, agent.steps_per_update)
        stats = agent.update(states, actions, adv, rets, logps)
        eqs = [i["equity"] for i in infos] or [env.equity]
        avg_eq = float(np.mean(eqs))
        history.append(avg_eq)
        best_equity = max(best_equity, avg_eq)
        if (up + 1) % 10 == 0 or up == 0:
            log(f"PPO update {up + 1}/{total_updates} | avg final equity {avg_eq:.4f} | pi {stats['pi_loss']:.4f} v {stats['v_loss']:.4f} H {stats['entropy']:.3f}")
    log(f"PPO training finished in {time.time() - t0:.1f}s")
    return history
