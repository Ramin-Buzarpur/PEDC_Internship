from __future__ import annotations

from pathlib import Path
from typing import Iterable, Optional, Sequence

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, Subset


FEATURE_NAMES = (
    "open", "high", "low", "close", "volume", "return",
    "volatility", "momentum", "volume_change", "range",
)


class UnifiedMarketDataset(Dataset):
    """Chronological multi-asset windows from CSV files or synthetic demo data.

    Expected CSV columns are Date, Open, High, Low, Close and Volume. Files may
    be directly under data_dir or under data_dir/<market>/csv.
    """

    def __init__(
        self,
        data_dir: Optional[str] = None,
        markets: Sequence[str] = ("nasdaq", "nyse", "sp500", "forbes2000"),
        seq_len: int = 30,
        future_len: int = 5,
        max_stocks: int = 12,
        corr_threshold: float = 0.35,
        synthetic_days: int = 220,
        seed: int = 42,
    ) -> None:
        self.seq_len = seq_len
        self.future_len = future_len
        frames = self._load_csv_frames(data_dir, markets, max_stocks)
        self.is_synthetic = not frames
        if self.is_synthetic:
            frames = self._make_synthetic_frames(max_stocks, synthetic_days, seed)
        self.symbols, aligned = self._align_frames(frames)
        if len(aligned) <= seq_len + future_len:
            raise ValueError("Not enough aligned rows for the requested windows")

        raw_returns = aligned.xs("Close", axis=1, level=1).pct_change().fillna(0.0)
        feature_blocks = [self._features(aligned[symbol]) for symbol in self.symbols]
        values = np.stack([frame.to_numpy(np.float32) for frame in feature_blocks], axis=1)
        split_at = max(seq_len, int(len(values) * 0.7))
        mean = values[:split_at].mean(axis=0, keepdims=True)
        std = values[:split_at].std(axis=0, keepdims=True) + 1e-6
        self.data = np.nan_to_num((values - mean) / std, nan=0.0, posinf=0.0, neginf=0.0)
        self.returns = raw_returns.to_numpy(np.float32)
        self.graph = self._build_graph(self.returns[:split_at], corr_threshold)
        self.num_features = len(FEATURE_NAMES)
        self.length = len(self.data) - seq_len - future_len + 1

    @staticmethod
    def _load_csv_frames(data_dir, markets, max_stocks):
        if not data_dir:
            return {}
        root = Path(data_dir).expanduser()
        candidates = list(root.glob("*.csv"))
        for market in markets:
            candidates.extend((root / market / "csv").glob("*.csv"))
        frames = {}
        for path in sorted(set(candidates))[:max_stocks]:
            try:
                frame = pd.read_csv(path)
                date_name = next((c for c in ("Date", "date", "Timestamp") if c in frame), None)
                required = ["Open", "High", "Low", "Close", "Volume"]
                if date_name is None or not all(c in frame for c in required):
                    continue
                frame["Date"] = pd.to_datetime(frame[date_name], errors="coerce", utc=True)
                frame = frame.dropna(subset=["Date"]).set_index("Date").sort_index()
                numeric = frame[required].apply(pd.to_numeric, errors="coerce")
                frames[path.stem] = numeric[~numeric.index.duplicated(keep="last")]
            except (OSError, ValueError, pd.errors.ParserError):
                continue
        return frames

    @staticmethod
    def _make_synthetic_frames(num_stocks, days, seed):
        rng = np.random.default_rng(seed)
        dates = pd.date_range("2020-01-01", periods=days, freq="B", tz="UTC")
        market = rng.normal(0.0003, 0.008, days)
        frames = {}
        for idx in range(max(2, num_stocks)):
            returns = 0.55 * market + rng.normal(0.0001, 0.01, days)
            close = 100.0 * np.exp(np.cumsum(returns))
            open_ = close * (1 + rng.normal(0, 0.002, days))
            high = np.maximum(open_, close) * (1 + rng.uniform(0, 0.006, days))
            low = np.minimum(open_, close) * (1 - rng.uniform(0, 0.006, days))
            volume = rng.lognormal(13.0, 0.35, days)
            frames[f"SYN{idx:03d}"] = pd.DataFrame(
                {"Open": open_, "High": high, "Low": low, "Close": close, "Volume": volume},
                index=dates,
            )
        return frames

    @staticmethod
    def _align_frames(frames):
        symbols = sorted(frames)
        aligned = pd.concat({s: frames[s] for s in symbols}, axis=1, join="inner").ffill().dropna()
        return symbols, aligned

    @staticmethod
    def _features(df):
        close = df["Close"].clip(lower=1e-8)
        volume = df["Volume"].clip(lower=1.0)
        ret = np.log(close / close.shift(1))
        result = pd.DataFrame(index=df.index)
        result["open"] = np.log(df["Open"].clip(lower=1e-8) / close)
        result["high"] = np.log(df["High"].clip(lower=1e-8) / close)
        result["low"] = np.log(df["Low"].clip(lower=1e-8) / close)
        result["close"] = ret.rolling(5, min_periods=1).mean()
        result["volume"] = np.log1p(volume)
        result["return"] = ret
        result["volatility"] = ret.rolling(5, min_periods=2).std()
        result["momentum"] = close.pct_change(10)
        result["volume_change"] = volume.pct_change(5)
        result["range"] = (df["High"] - df["Low"]) / close
        return result.replace([np.inf, -np.inf], np.nan).fillna(0.0)

    @staticmethod
    def _build_graph(returns, threshold):
        corr = np.nan_to_num(np.corrcoef(returns.T), nan=0.0)
        positive = (corr >= threshold).astype(np.float32)
        negative = (corr <= -threshold).astype(np.float32)
        np.fill_diagonal(positive, 1.0)
        return torch.from_numpy(np.stack([positive, negative], axis=-1))

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        if index < 0 or index >= self.length:
            raise IndexError(index)
        end = index + self.seq_len
        past = torch.from_numpy(self.data[index:end]).permute(1, 2, 0).contiguous()
        future = torch.from_numpy(self.returns[end:end + self.future_len]).T.contiguous()
        return past, future, self.graph


def chronological_split(dataset: Dataset, fractions=(0.7, 0.15, 0.15)):
    if not np.isclose(sum(fractions), 1.0):
        raise ValueError("Split fractions must sum to 1")
    n = len(dataset)
    train_end = int(n * fractions[0])
    val_end = train_end + int(n * fractions[1])
    if train_end == 0 or val_end == train_end or val_end >= n:
        raise ValueError("Dataset is too small for a three-way chronological split")
    return (
        Subset(dataset, range(0, train_end)),
        Subset(dataset, range(train_end, val_end)),
        Subset(dataset, range(val_end, n)),
    )
