from __future__ import annotations

import numpy as np
import torch

from src.models.rl_agent import PPOAgent, PortfolioEnv, collect_rollout, gae


def make_env(T=60, N=4, start=0, end=58):
    rng = np.random.default_rng(5)
    rets = rng.normal(0.0005, 0.01, size=(T, N)).astype(np.float32)
    feats = rng.normal(0, 1, size=(T, N, 3)).astype(np.float32)
    vol = rng.uniform(0.008, 0.03, size=(T, N)).astype(np.float32)
    return PortfolioEnv(rets, feats, vol, start=start, end=end, cost_bps=5.0)


def test_env_state_dim_and_step():
    env = make_env()
    s = env.reset()
    assert s.shape == (env.state_dim,)
    a = np.zeros(env.n_assets)
    s2, r, done, info = env.step(a)
    assert isinstance(r, float) and np.isfinite(r)
    assert "port_ret" in info


def test_env_respects_leverage_and_weight_caps():
    env = make_env()
    env.reset()
    big_action = np.full(env.n_assets, 10.0)
    w = env._target_weights(big_action)
    assert np.abs(w).max() <= env.max_weight + 1e-8
    assert np.abs(w).sum() <= env.max_leverage + 1e-8


def test_vol_targeting_scales_positions():
    env = make_env(T=60, N=4, start=0, end=58)
    env.reset()
    action = np.ones(env.n_assets) * 3.0
    w = env._target_weights(action)
    implied = w / (env.vol_target_daily / env.vol[0])
    uncapped = ~((np.abs(w) >= env.max_weight - 1e-6) | (np.abs(w).sum() >= env.max_leverage - 1e-6))
    if uncapped.any():
        assert np.allclose(implied[uncapped], np.tanh(3.0), atol=0.02)


def test_vol_targeting_caps_extreme_actions():
    env = make_env(T=60, N=4, start=0, end=58)
    env.reset()
    env.max_weight = 10.0
    env.max_leverage = 100.0
    w = env._target_weights(np.full(env.n_assets, 5.0))
    assert np.allclose(w, np.tanh(5.0) * env.vol_target_daily / env.vol[0], atol=0.02)
    env2 = make_env(T=60, N=4, start=0, end=58)
    env2.reset()
    w2 = env2._target_weights(np.full(env2.n_assets, 100.0))
    assert np.abs(w2).max() <= env2.max_weight + 1e-8


def test_base_signal_residual():
    env = make_env(T=60, N=4, start=0, end=58)
    env.max_weight = 10.0
    env.max_leverage = 100.0
    base = np.full((60, 4), 0.1, dtype=np.float32)
    env.base_signal = base
    env.resid_scale = 1.0
    env.t = 10
    w_zero_action = env._target_weights(np.zeros(4))
    assert np.allclose(w_zero_action, 0.1 * env.vol_target_daily / env.vol[10], atol=1e-6)
    w_pos = env._target_weights(np.full(4, 3.0))
    expected = (0.1 + np.tanh(3.0)) * env.vol_target_daily / env.vol[10]
    assert np.allclose(w_pos, expected, atol=1e-5)


def test_gae_shapes_and_finiteness():
    rewards = [0.1, -0.2, 0.05]
    values = [0.0, 0.1, 0.2]
    dones = [False, False, True]
    adv = gae(rewards, values, dones, last_value=0.0, gamma=0.99, lam=0.95)
    assert len(adv) == 3 and np.isfinite(adv).all()


def test_ppo_update_runs_and_improves_logprob_finite():
    torch.manual_seed(0)
    env = make_env(T=40, end=38)
    cfg = {"hidden_dim": 32, "layers": 1, "lr": 1e-3, "steps_per_update": 64}
    agent = PPOAgent(env.state_dim, env.n_assets, cfg, device="cpu")
    states, actions, adv, rets, logps, infos = collect_rollout(env, agent, 64)
    assert len(states) == 64
    stats = agent.update(states, actions, adv, rets, logps)
    assert all(np.isfinite(v) for v in stats.values())
