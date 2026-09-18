# Data directory

Real data is intentionally not included. Put CSV files either directly here or in:

`data/<market>/csv/<symbol>.csv`

Each file requires: `Date, Open, High, Low, Close, Volume`.

Run `python main.py --data-dir data`. If no valid CSV is found, the application
uses reproducible synthetic data and clearly reports `data_mode: synthetic`.
