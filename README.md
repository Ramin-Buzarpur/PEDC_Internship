# PEDC Internship

This repository holds all of my work from the internship at Tafahom Company (summer 2026). The topic is forecasting and trading financial markets with diffusion models and reinforcement learning, plus several research lines on loss functions and time-series geometry. All code, outputs and results are here, including the ones that did not work.

> **Disclaimer:** This is a research and learning repository, not financial advice. Backtest results do not guarantee future performance.

## Results at a glance

I report results as they came out. Many of them are negative or neutral, and I think that is the real value of the work.

| Work line | Main result |
|---|---|
| **PEDC Alpha** (trading system) | Walk-forward backtest over 2,143 trading days: CAGR 12.9% and Sharpe 0.87, versus 18.6% and 0.95 for buy-and-hold. Drawdown was smaller (27.1% vs 32.5%). The "alpha" sleeve (diffusion + RL) lost money after costs (CAGR -3.4%, Sharpe -0.43). |
| **SC-twCRPS** (new loss) | On GARCH data, version v0.4 was worse than plain CRPS (0.572 vs 0.548). The first idea did not work. |
| **SAFPS** (risk-aware loss weighting) | On a 30-seed holdout, Monotonic Hybrid improved tail calibration gap in 29 of 30 seeds (about 30% lower on the stress tail), at the cost of slightly worse overall CRPS. |
| **MLDY** (oil forecasting with Finsler geometry) | Interval coverage of only 27.4%, versus about 81% for simple statistical methods. It needs calibration before any use. |

## Repository layout

| Path | Description |
|---|---|
| [`engine/`](./engine) | Clean PEDC Alpha core: data, features, MaTCHS diffusion, RL agent, backtest and 37 unit tests. Details in [`engine/README.md`](./engine/README.md). |
| [`research/crps-loss/`](./research/crps-loss) | Loss-function research: `SC_twCRPS.py`, `SAFPS.py`, diagnostic scripts `_diag*`, `sanity_check.py` and the full story in `research_log.md`. |
| [`safps_*`](.) | Outputs of the SAFPS versions (v0.12 to v4) as CSV files and result folders. |
| [`MLDY/`](./MLDY) | Brent oil forecasting engine, five statistical baselines, ablations and history-length sensitivity. |
| [`dashboard/`](./dashboard) | Web dashboard (React + Vite) for comparing models. |
| [`docs/`](./docs) | Reference papers and images. |
| [`archive/`](./archive) | Older and parallel versions (`fin_gemini`, `fin_deepseek`, `main_diffstock`, AlphaMarketAI). Kept for reference only. |

## Quick start

### PEDC Alpha

```bash
cd engine
pip install -r requirements.txt
python3 scripts/prepare_cache.py     # prepare data for the 4 markets
python3 scripts/run_training.py      # train diffusion and RL
python3 scripts/run_backtest.py      # walk-forward backtest
python3 -m pytest tests/ -q          # tests
```

### CRPS / SAFPS research

```bash
cd research/crps-loss
python3 sanity_check.py              # sanity check before each experiment
python3 SAFPS.py
```

The full results and the reasoning behind the experiments are recorded in `research_log.md`. A few of the latest experiments there are still marked "pending" and no result is claimed for them.

### MLDY

```bash
cd MLDY
python3 build_dataset.py             # build the oil geodesic dataset
python3 compare_methods.py           # compare against statistical methods
```

## Evaluation protocol

- **Backtest:** walk-forward with 5 bps commission and 2 bps slippage. The whole system's break-even cost is 48 bps.
- **SAFPS:** seeds are split into three groups (tuning, verification, final holdout). 95% confidence intervals use bootstrap with 10,000 resamples, and all comparisons are paired.
- **MLDY:** the number of evaluation samples differs across methods (1,662 for the simple ones, 175 for Finsler, 793 for GARCH), so read the n column next to every number.

## Known limitations

- The news archive only covers the last 14 days, so sentiment is zero in the historical backtest.
- Delisted symbols continue with a flat price, and the universe should be refreshed with `prepare_cache`.
- SAFPS v4 has no holdout, so its results are not a basis for any claim.
- TCM-CRPS, EHF-CRPS and SQL-CRPS are at the design and partial-code stage and have no claimable results.

## Reference papers

The main papers (DiffStock, DiffSTG, Adv-ALSTM and several time-series papers) are in [`docs/papers`](./docs/papers).

## Author

Ramin Buzarpur, Tafahom Company internship, summer 2026.