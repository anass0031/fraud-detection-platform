"""One call -> every metric, overall and excluding uids that already had a fraud in train.

The threshold is chosen on the overall population and then *reused* for the excluding subset,
so both rows describe the same alert rule.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from fraudml.evaluate import metrics as m
from fraudml.evaluate.cost import CostConfig, cost_breakdown
from fraudml.evaluate.threshold import optimise_threshold


def evaluate_scores(
    y,
    score,
    amount,
    day,
    cfg: CostConfig | None = None,
    threshold: float | None = None,
    is_probability: bool = True,
) -> dict[str, float]:
    """All metrics for one population. If `threshold` is None, the cost-optimal one is chosen."""
    cfg = cfg or CostConfig()
    y = np.asarray(y).astype(int)
    score = np.asarray(score, dtype="float64")
    amount = np.asarray(amount, dtype="float64")
    day = np.asarray(day)
    k = cfg.capacity_per_day

    if threshold is None:
        threshold = optimise_threshold(y, score, amount, day, cfg).threshold
    alert = score >= threshold
    alerts_per_day = pd.Series(alert).groupby(day).sum()
    topk = m.daily_topk_mask(score, day, k)

    out: dict[str, float] = {
        "n": int(len(y)),
        "fraud_rate": float(y.mean()),
        "pr_auc": m.pr_auc(y, score),
        "roc_auc": m.roc_auc(y, score),
        "brier": m.brier(y, score) if is_probability else float("nan"),
        "recall_at_p50": m.recall_at_precision(y, score, 0.5),
        "precision_at_capacity": m.precision_at_capacity(y, score, day, k),
        "recall_at_capacity": m.recall_at_capacity(y, score, day, k),
        "fraud_value_caught_at_capacity": m.fraud_value_caught(y, amount, topk),
        "threshold": float(threshold),
        "alerts_per_day": float(alerts_per_day.mean()),
        "max_alerts_per_day": int(alerts_per_day.max()),
        "precision_at_threshold": float(y[alert].mean()) if alert.any() else 0.0,
        "recall_at_threshold": float(y[alert].sum() / y.sum()) if y.sum() else 0.0,
        "fraud_value_caught_at_threshold": m.fraud_value_caught(y, amount, alert),
    }
    out.update({key: float(v) for key, v in cost_breakdown(y, amount, alert, cfg).items()})
    return out


def evaluate_report(
    y,
    score,
    amount,
    day,
    fraud_uid_in_train,
    cfg: CostConfig | None = None,
    threshold: float | None = None,
    is_probability: bool = True,
) -> pd.DataFrame:
    """Two rows: `overall` and `excl_fraud_uids` (rows whose uid had a fraud in train removed)."""
    cfg = cfg or CostConfig()
    overall = evaluate_scores(y, score, amount, day, cfg, threshold, is_probability)
    keep = ~np.asarray(fraud_uid_in_train).astype(bool)
    sub = [np.asarray(a)[keep] for a in (y, score, amount, day)]
    excl = evaluate_scores(*sub, cfg, overall["threshold"], is_probability)
    index = pd.Index(["overall", "excl_fraud_uids"], name="population")
    return pd.DataFrame([overall, excl], index=index)
