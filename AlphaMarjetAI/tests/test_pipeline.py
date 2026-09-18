import torch

from alphamarketai.datasets import UnifiedMarketDataset, chronological_split
from alphamarketai.decision import TradingPolicy
from alphamarketai.models import AlphaMaTCHS, DiffusionDenoiser, GaussianDiffusion


def test_dataset_and_chronological_split():
    dataset = UnifiedMarketDataset(seq_len=12, future_len=3, max_stocks=3, synthetic_days=60)
    past, future, graph = dataset[0]
    assert past.shape == (3, 10, 12)
    assert future.shape == (3, 3)
    assert graph.shape == (3, 3, 2)
    train, validation, test = chronological_split(dataset)
    assert max(train.indices) < min(validation.indices) < min(test.indices)


def test_end_to_end_tensor_contract():
    dataset = UnifiedMarketDataset(seq_len=12, future_len=3, max_stocks=3, synthetic_days=60)
    past, future, graph = dataset[0]
    model = AlphaMaTCHS(features=10, hidden=16, heads=4)
    output = model(past.unsqueeze(0), graph)
    assert output["return"].shape == (1, 3)
    denoiser = DiffusionDenoiser(future_len=3, condition_dim=16, hidden=16)
    diffusion = GaussianDiffusion(denoiser, steps=3)
    loss = diffusion.training_loss(future, output["embedding"].squeeze(0))
    assert torch.isfinite(loss)
    scenarios = diffusion.sample(output["embedding"], num_samples=4)
    decision = TradingPolicy().decide(scenarios)
    assert scenarios.shape == (1, 3, 4, 3)
    assert decision.action.shape == (1, 3)
