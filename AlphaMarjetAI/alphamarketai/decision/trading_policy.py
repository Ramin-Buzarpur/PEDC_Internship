from dataclasses import dataclass

import torch


@dataclass
class TradingDecision:
    action: torch.Tensor
    score: torch.Tensor
    confidence: torch.Tensor
    expected_return: torch.Tensor
    downside_risk: torch.Tensor


class TradingPolicy:
    """Risk-aware BUY(+1), HOLD(0), SELL(-1) policy from forecast scenarios."""
    def __init__(self, buy_threshold=0.001, sell_threshold=-0.001, risk_aversion=0.5):
        self.buy_threshold = buy_threshold
        self.sell_threshold = sell_threshold
        self.risk_aversion = risk_aversion

    def decide(self, scenarios):
        terminal = scenarios[..., -1]
        expected = terminal.mean(dim=-1)
        downside = terminal.clamp(max=0).square().mean(dim=-1).sqrt()
        score = expected - self.risk_aversion * downside
        action = torch.zeros_like(score, dtype=torch.int64)
        action[score > self.buy_threshold] = 1
        action[score < self.sell_threshold] = -1
        confidence = torch.where(action >= 0, (terminal > 0).float().mean(-1), (terminal < 0).float().mean(-1))
        return TradingDecision(action, score, confidence, expected, downside)
