import numpy as np
import torch


class WalkForwardBacktest:
    def __init__(self, transaction_cost=0.001, annual_factor=252):
        self.transaction_cost = transaction_cost
        self.annual_factor = annual_factor

    @torch.no_grad()
    def run(self, model, dataset, device):
        model.eval()
        daily_returns, equity = [], []
        capital = 1.0
        previous_weights = None
        for past, future, graph in dataset:
            output = model(past.unsqueeze(0).to(device), graph.to(device))
            scores = torch.tanh(output["direction"])[0].cpu().numpy()
            gross_exposure = np.abs(scores).sum()
            weights = scores / gross_exposure if gross_exposure > 1e-8 else np.zeros_like(scores)
            realized = future[:, 0].numpy()
            turnover = np.abs(weights - previous_weights).sum() if previous_weights is not None else np.abs(weights).sum()
            net_return = float(np.dot(weights, realized) - turnover * self.transaction_cost)
            capital *= max(1e-8, 1.0 + net_return)
            daily_returns.append(net_return); equity.append(capital); previous_weights = weights
        returns = np.asarray(daily_returns)
        curve = np.asarray(equity)
        running_max = np.maximum.accumulate(curve)
        volatility = returns.std(ddof=1) if len(returns) > 1 else 0.0
        return {
            "total_return": float(curve[-1] - 1.0),
            "sharpe": float(returns.mean() / (volatility + 1e-8) * np.sqrt(self.annual_factor)),
            "max_drawdown": float((curve / running_max - 1.0).min()),
            "win_rate": float((returns > 0).mean()),
            "observations": int(len(returns)),
        }
