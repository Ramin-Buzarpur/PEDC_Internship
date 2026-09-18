import argparse
import json

import torch
from torch.utils.data import DataLoader

from alphamarketai.backtest import WalkForwardBacktest
from alphamarketai.configs import AppConfig
from alphamarketai.datasets import UnifiedMarketDataset, chronological_split
from alphamarketai.decision import TradingPolicy
from alphamarketai.models import AlphaMaTCHS, DiffusionDenoiser, GaussianDiffusion
from alphamarketai.training import MarketTrainer
from alphamarketai.utils import seed_everything


def parse_args():
    parser = argparse.ArgumentParser(description="Train and evaluate AlphaMarketAI")
    parser.add_argument("--data-dir", default=None, help="CSV root; omit for deterministic synthetic demo")
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-stocks", type=int, default=12)
    parser.add_argument("--synthetic-days", type=int, default=220)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--quick", action="store_true", help="Small CPU smoke run")
    return parser.parse_args()


def resolve_device(value):
    if value == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    return torch.device("cuda" if value == "auto" and torch.cuda.is_available() else ("cpu" if value == "auto" else value))


def main():
    args = parse_args()
    config = AppConfig()
    config.data.data_dir = args.data_dir
    config.data.max_stocks = 4 if args.quick else args.max_stocks
    config.data.synthetic_days = 100 if args.quick else args.synthetic_days
    config.training.epochs = 1 if args.quick else args.epochs
    config.training.batch_size = min(8, args.batch_size) if args.quick else args.batch_size
    config.model.hidden = 16 if args.quick else config.model.hidden
    config.model.diffusion_steps = 5 if args.quick else config.model.diffusion_steps
    config.validate(); seed_everything(config.data.seed)
    device = resolve_device(args.device)

    dataset = UnifiedMarketDataset(**vars(config.data))
    train_set, val_set, test_set = chronological_split(dataset)
    train_loader = DataLoader(train_set, batch_size=config.training.batch_size, shuffle=False,
                              num_workers=config.training.num_workers, drop_last=False)
    val_loader = DataLoader(val_set, batch_size=config.training.batch_size, shuffle=False,
                            num_workers=config.training.num_workers, drop_last=False)
    model = AlphaMaTCHS(config.model.features, config.model.hidden, config.model.heads)
    denoiser = DiffusionDenoiser(config.data.future_len, config.model.hidden, config.model.hidden * 2)
    diffusion = GaussianDiffusion(denoiser, config.model.diffusion_steps)
    trainer = MarketTrainer(model, diffusion, device, config.training.learning_rate,
                            config.training.weight_decay, config.training.grad_clip)
    checkpoint = config.output_path / "checkpoints" / "market_ai.pt"
    trainer.fit(train_loader, config.training.epochs, checkpoint, val_loader)

    result = WalkForwardBacktest().run(model, test_set, device)
    past, _, graph = test_set[-1]
    model.eval(); diffusion.eval()
    with torch.no_grad():
        output = model(past.unsqueeze(0).to(device), graph.to(device))
        scenarios = diffusion.sample(output["embedding"], num_samples=16)
        decision = TradingPolicy().decide(scenarios)
    result["device"] = str(device)
    result["data_mode"] = "synthetic" if dataset.is_synthetic else "csv"
    result["latest_actions"] = {
        symbol: {"action": {1: "BUY", 0: "HOLD", -1: "SELL"}[int(decision.action[0, i])],
                 "confidence": round(float(decision.confidence[0, i]), 4)}
        for i, symbol in enumerate(dataset.symbols)
    }
    config.output_path.mkdir(parents=True, exist_ok=True)
    (config.output_path / "metrics.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
