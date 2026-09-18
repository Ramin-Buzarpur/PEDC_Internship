from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import load_config
from src.pipeline import prepare_data
from src.utils.runtime import get_logger


def main():
    ap = argparse.ArgumentParser(description="Pre-build the processed data cache (slow first scan, fast later runs)")
    ap.add_argument("--config", default=None)
    ap.add_argument("--stocks", type=int, default=None)
    ap.add_argument("--rescan", action="store_true")
    args = ap.parse_args()

    cfg = load_config(args.config)
    if args.stocks:
        cfg.raw["data"]["max_stocks"] = args.stocks
    log = get_logger("cache")
    dates, symbols, arrays, _ = prepare_data(cfg, fetch_news=False, log=log.info)
    log.info(f"cache ready: T={len(dates)} N={len(symbols)} range {dates[0].date()} → {dates[-1].date()}")


if __name__ == "__main__":
    main()
