"""Cost-optimal threshold under the daily review capacity.

Alert rule: score >= threshold. Among thresholds whose mean alerts per day is within the
capacity, pick the one with the lowest total cost. "No alerts" (threshold = +inf) is always a
candidate, so a feasible answer always exists.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from fraudml.evaluate.cost import CostConfig


@dataclass(frozen=True)
class ThresholdResult:
    threshold: float
    total_cost: float
    alerts_per_day: float  # mean over days
    max_alerts_per_day: int
    n_alerts: int


def cost_curve(
    y, score, amount, day, cfg: CostConfig | None = None, n_grid: int = 1000
) -> pd.DataFrame:
    """Total cost and mean alerts/day for a grid of thresholds (score quantiles + no-alert)."""
    cfg = cfg or CostConfig()
    y = np.asarray(y).astype(bool)
    score = np.asarray(score, dtype="float64")
    amount = np.asarray(amount, dtype="float64")
    n_days = len(np.unique(np.asarray(day)))

    grid = np.unique(score)
    if len(grid) > n_grid:  # large data: score quantiles are enough
        grid = np.unique(np.quantile(score, np.linspace(0.0, 1.0, n_grid)))
    grid = np.append(grid, np.inf)

    # Sort descending once; the alerts for threshold t are the first n rows with score >= t.
    order = np.argsort(-score, kind="stable")
    neg_sorted = -score[order]  # ascending
    cum_tp = np.concatenate([[0], np.cumsum(y[order])])
    caught = np.where(y[order], amount[order], 0.0)
    cum_caught = np.concatenate([[0.0], np.cumsum(caught)])
    n_alerts = np.searchsorted(neg_sorted, -grid, side="right")

    tp = cum_tp[n_alerts]
    fp = n_alerts - tp
    missed = amount[y].sum() - cum_caught[n_alerts]
    total = missed + n_alerts * cfg.review_cost + fp * cfg.friction_cost
    per_day = n_alerts / n_days
    return pd.DataFrame(
        {
            "threshold": grid,
            "n_alerts": n_alerts,
            "tp": tp,
            "fp": fp,
            "alerts_per_day": per_day,
            "total_cost": total,
            "feasible": per_day <= cfg.capacity_per_day,
        }
    )


def optimise_threshold(
    y, score, amount, day, cfg: CostConfig | None = None, n_grid: int = 1000
) -> ThresholdResult:
    cfg = cfg or CostConfig()
    curve = cost_curve(y, score, amount, day, cfg, n_grid)
    best = curve[curve["feasible"]].sort_values(["total_cost", "threshold"]).iloc[0]
    t = float(best["threshold"])
    alerts = np.asarray(score, dtype="float64") >= t
    per_day = pd.Series(alerts).groupby(np.asarray(day)).sum()
    return ThresholdResult(
        threshold=t,
        total_cost=float(best["total_cost"]),
        alerts_per_day=float(best["alerts_per_day"]),
        max_alerts_per_day=int(per_day.max()),
        n_alerts=int(best["n_alerts"]),
    )
