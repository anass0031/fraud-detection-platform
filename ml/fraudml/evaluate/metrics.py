"""Ranking and capacity metrics.

All functions take plain arrays: y (0/1), score (higher = riskier), and when needed the
transaction amount and the day index. Capacity metrics rank alerts *within each day*,
because analysts can review at most `k` cases per day.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    precision_recall_curve,
    roc_auc_score,
)


def pr_auc(y, score) -> float:
    """Area under the precision-recall curve (average precision). Random ranking = fraud rate."""
    return float(average_precision_score(np.asarray(y), np.asarray(score)))


def roc_auc(y, score) -> float:
    return float(roc_auc_score(np.asarray(y), np.asarray(score)))


def brier(y, prob) -> float:
    """Mean squared error of probabilities. Only meaningful if `prob` is a probability."""
    return float(brier_score_loss(np.asarray(y), np.clip(np.asarray(prob), 0.0, 1.0)))


def recall_at_precision(y, score, min_precision: float = 0.5) -> float:
    """Highest recall reachable with precision >= min_precision (0 if never reached)."""
    precision, recall, _ = precision_recall_curve(np.asarray(y), np.asarray(score))
    ok = precision >= min_precision
    return float(recall[ok].max()) if ok.any() else 0.0


def daily_topk_mask(score, day, k: int) -> np.ndarray:
    """True for the k highest scores of each day (ties broken by original order)."""
    s = pd.Series(np.asarray(score, dtype="float64"))
    rank = s.groupby(np.asarray(day)).rank(method="first", ascending=False)
    return (rank <= k).to_numpy()


def precision_at_capacity(y, score, day, k: int) -> float:
    """Precision of the daily top-k alerts, pooled over days (fraud alerts / all alerts)."""
    mask = daily_topk_mask(score, day, k)
    return float(np.asarray(y)[mask].mean()) if mask.any() else 0.0


def recall_at_capacity(y, score, day, k: int) -> float:
    """Share of fraud *cases* inside the daily top-k alerts."""
    y = np.asarray(y)
    mask = daily_topk_mask(score, day, k)
    return float(y[mask].sum() / y.sum()) if y.sum() else 0.0


def fraud_value_caught(y, amount, alert_mask) -> float:
    """Share of fraud *value* ($) that is inside the alerts."""
    y = np.asarray(y).astype(bool)
    amount = np.asarray(amount, dtype="float64")
    total = amount[y].sum()
    caught = amount[y & np.asarray(alert_mask).astype(bool)].sum()
    return float(caught / total) if total > 0 else 0.0
