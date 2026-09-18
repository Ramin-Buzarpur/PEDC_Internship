from __future__ import annotations

import numpy as np
import pandas as pd

from src.data.news import daily_sentiment_array, score_headline


def test_positive_headline_scores_positive():
    assert score_headline("Apple beats earnings expectations, raises guidance") > 0.1


def test_negative_headline_scores_negative():
    assert score_headline("Company plunges amid fraud investigation and layoffs") < -0.1


def test_neutral_headline_near_zero():
    assert abs(score_headline("The company announced its quarterly calendar")) <= 0.15


def test_negation_flips_sentiment():
    plain = score_headline("Stock surges on record profits")
    negated = score_headline("Stock denies surge rumors")
    assert negated <= plain


def test_daily_sentiment_alignment_and_decay():
    dates = pd.bdate_range("2024-01-01", periods=10)
    news = {"AAA": [{"date": "2024-01-05", "title": "surge", "score": 1.0}]}
    arr = daily_sentiment_array(news, ["AAA"], dates, half_life_days=2.0)
    k = list(dates).index(pd.Timestamp("2024-01-05"))
    assert arr[k, 0] == 1.0
    assert 0 < arr[k + 1, 0] < arr[k, 0]
    assert arr[k + 3, 0] < arr[k + 1, 0]
    assert np.abs(arr).max() <= 1.0


def test_sentiment_ignores_unknown_dates():
    dates = pd.bdate_range("2024-01-01", periods=5)
    news = {"BBB": [{"date": "2030-01-01", "title": "x", "score": 0.9}]}
    arr = daily_sentiment_array(news, ["BBB"], dates)
    assert (arr == 0).all()
