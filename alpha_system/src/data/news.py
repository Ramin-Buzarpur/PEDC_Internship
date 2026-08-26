from __future__ import annotations

import json
import os
import re
import time
import urllib.parse
import urllib.request
from datetime import datetime, timedelta, timezone
from xml.etree import ElementTree

import numpy as np

POSITIVE_WORDS = {
    "beat": 1.0, "beats": 1.0, "surge": 1.0, "surges": 1.0, "soar": 1.2, "soars": 1.2, "jump": 0.8,
    "jumps": 0.8, "rally": 1.0, "rallies": 1.0, "gain": 0.6, "gains": 0.6, "upgrade": 1.2, "upgraded": 1.2,
    "outperform": 1.0, "record": 0.8, "strong": 0.7, "growth": 0.5, "profit": 0.7, "profits": 0.7,
    "bullish": 1.2, "buy": 0.6, "buys": 0.4, "raise": 0.5, "raises": 0.5, "tops": 0.8, "boost": 0.7,
    "boosts": 0.7, "dividend": 0.4, "expansion": 0.4, "breakthrough": 0.9, "approval": 0.8, "approved": 0.8,
    "partnership": 0.5, "acquisition": 0.3, "wins": 0.7, "win": 0.6, "optimistic": 0.8, "rebound": 0.7,
    "recover": 0.6, "recovery": 0.6, "high-demand": 0.6, "momentum": 0.4, "upside": 0.7, "overweight": 0.9,
}
NEGATIVE_WORDS = {
    "miss": -1.0, "misses": -1.0, "missed": -1.0, "plunge": -1.2, "plunges": -1.2, "slump": -1.0,
    "slumps": -1.0, "drop": -0.7, "drops": -0.7, "fall": -0.6, "falls": -0.6, "decline": -0.6,
    "downgrade": -1.2, "downgraded": -1.2, "underperform": -1.0, "loss": -0.8, "losses": -0.8,
    "bearish": -1.2, "sell": -0.5, "sells": -0.4, "cut": -0.6, "cuts": -0.6, "lawsuit": -0.8,
    "probe": -0.8, "investigation": -0.9, "fraud": -1.5, "recall": -0.9, "bankruptcy": -1.8,
    "layoff": -1.0, "layoffs": -1.0, "warns": -0.9, "warning": -0.8, "weak": -0.7, "slowdown": -0.9,
    "recession": -1.3, "crash": -1.5, "tumble": -1.0, "tumbles": -1.0, "slide": -0.8, "risk": -0.4,
    "concern": -0.5, "concerns": -0.5, "shortfall": -0.9, "halt": -0.7, "delisting": -1.4,
    "sec": -0.4, "fine": -0.5, "fined": -0.6, "penalty": -0.7, "default": -1.4, "selloff": -1.1,
}
WORD_RE = re.compile(r"[a-zA-Z]{3,}")


def score_headline(title: str) -> float:
    tokens = WORD_RE.findall(title.lower())
    if not tokens:
        return 0.0
    s = 0.0
    for i, tok in enumerate(tokens):
        negated = i > 0 and tokens[i - 1] in {"not", "no", "without", "denies", "deny"}
        val = POSITIVE_WORDS.get(tok, 0.0) + NEGATIVE_WORDS.get(tok, 0.0)
        if val and negated:
            val *= -0.8
        s += val
    return float(np.tanh(s / max(3.0, np.sqrt(len(tokens)))))


def _fetch_google_news(symbol: str, max_items: int = 15, timeout: int = 12) -> list[dict]:
    query = urllib.parse.quote(f"{symbol} stock")
    url = f"https://news.google.com/rss/search?q={query}+when:14d&hl=en-US&gl=US&ceid=US:en"
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            data = r.read()
        root = ElementTree.fromstring(data)
    except Exception:
        return []
    items: list[dict] = []
    for item in root.iter("item"):
        title = (item.findtext("title") or "").strip()
        pub = (item.findtext("pubDate") or "").strip()
        if not title:
            continue
        try:
            dt = datetime.strptime(pub, "%a, %d %b %Y %H:%M:%S %z").astimezone(timezone.utc)
        except Exception:
            dt = datetime.now(timezone.utc)
        items.append({"date": dt.date().isoformat(), "title": title, "score": round(score_headline(title), 4)})
        if len(items) >= max_items:
            break
    return items


def fetch_and_cache_news(symbols: list[str], cache_dir: str, cache_days: int = 1, sleep: float = 0.4) -> dict[str, list[dict]]:
    news_dir = os.path.join(cache_dir, "news")
    os.makedirs(news_dir, exist_ok=True)
    now = time.time()
    out: dict[str, list[dict]] = {}
    for sym in symbols:
        path = os.path.join(news_dir, f"{sym}.json")
        if os.path.exists(path) and now - os.path.getmtime(path) < cache_days * 86400:
            try:
                with open(path, "r", encoding="utf-8") as f:
                    out[sym] = json.load(f)
                continue
            except Exception:
                pass
        items = _fetch_google_news(sym)
        out[sym] = items
        if items:
            with open(path, "w", encoding="utf-8") as f:
                json.dump(items, f, ensure_ascii=False)
        time.sleep(sleep)
    return out


def daily_sentiment_array(news: dict[str, list[dict]], symbols: list[str], dates, half_life_days: float = 2.0) -> np.ndarray:
    n_dates, n_syms = len(dates), len(symbols)
    arr = np.zeros((n_dates, n_syms), dtype=np.float32)
    date_pos = {d.strftime("%Y-%m-%d"): i for i, d in enumerate(dates)}
    decay = np.log(2.0) / max(half_life_days, 0.25)

    def to_dt(s: str) -> datetime | None:
        try:
            return datetime.strptime(s, "%Y-%m-%d")
        except Exception:
            return None

    for j, sym in enumerate(symbols):
        daily: dict[int, tuple[float, int]] = {}
        for it in news.get(sym, []):
            dt = to_dt(it["date"])
            if dt is None or dt.strftime("%Y-%m-%d") not in date_pos:
                continue
            k = date_pos[dt.strftime("%Y-%m-%d")]
            s_sum, cnt = daily.get(k, (0.0, 0))
            daily[k] = (s_sum + it["score"], cnt + 1)
        for k, (s_sum, cnt) in daily.items():
            avg = s_sum / cnt
            arr[k, j] += avg
            for fwd in range(1, min(10, n_dates - k)):
                arr[k + fwd, j] += avg * float(np.exp(-decay * fwd))
    return np.clip(arr, -1.0, 1.0)
