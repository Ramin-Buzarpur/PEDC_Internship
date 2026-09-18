# AlphaMarketAI

AlphaMarketAI is a research-oriented MVP for relational and probabilistic market
forecasting. It combines a compact MaTCHS-inspired temporal/graph model with a
conditional Gaussian diffusion model, a risk-aware BUY/SELL/HOLD policy, and a
chronological backtest.

> This is research software, not investment advice or a production trading bot.

## What was repaired

The received snapshot contained accidental duplicate files: `config.py` was a
copy of the backtest, `trainer.py` was a copy of the loss module, and
`diffusion_matchs.py` was a copy of `alpha_matchs.py`. The original random split
also leaked temporal ordering. This version replaces those modules and enforces
consistent tensor contracts throughout the pipeline.

## Quick start (no data required)

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python main.py --quick --device cpu
```

The quick command generates deterministic synthetic multi-asset OHLCV data,
trains for one epoch, runs the backtest, and writes:

- `artifacts/checkpoints/market_ai.pt`
- `artifacts/metrics.json`

## Run with real CSV data

```bash
python main.py --data-dir data --epochs 20 --batch-size 32
```

See `data/README.md` for the accepted layout. Normalization statistics and the
correlation graph are calculated only from the initial training period.

## Tests

```bash
pytest -q
```

## Scope

Implemented in this MVP:

- numerical OHLCV feature engineering;
- positive/negative inter-stock relation graph;
- temporal convolution and masked relational attention;
- deterministic return, direction and uncertainty heads;
- conditional DDPM future-return scenarios;
- risk-aware BUY/HOLD/SELL decisions;
- transaction-cost-aware chronological backtest.

News/sentiment ingestion, learned causal graphs, portfolio constraints, broker
execution, and paper/live trading are deliberately separate later milestones.
They require timestamped data contracts and leakage-safe evaluation before they
can be integrated responsibly.

## Research lineage

The architecture is informed by the supplied DiffStock/MaTCHS work, DiffSTG's
probabilistic graph forecasting, adversarial stock-movement training, and the
numerical-plus-textual forecasting literature. This code is an independent,
compact implementation and is not claimed to reproduce published results.
