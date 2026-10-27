"""Business cost model (idea.md §2).

TP: review cost · FP: review cost + friction · FN: transaction amount (lost money) · TN: 0.
"No model" baseline: every fraud is lost and nothing is reviewed, so its cost = total fraud value.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np


@dataclass(frozen=True)
class CostConfig:
    review_cost: float = 5.0  # $ per alert reviewed by an analyst
    friction_cost: float = 2.0  # $ per false alarm (customer friction)
    capacity_per_day: int = 150  # alerts analysts can review per day (3 x 50)

    def to_dict(self) -> dict:
        return asdict(self)


def cost_breakdown(y, amount, alert_mask, cfg: CostConfig | None = None) -> dict[str, float]:
    cfg = cfg or CostConfig()
    y = np.asarray(y).astype(bool)
    alert = np.asarray(alert_mask).astype(bool)
    amount = np.asarray(amount, dtype="float64")

    tp = int((y & alert).sum())
    fp = int((~y & alert).sum())
    fn = int((y & ~alert).sum())
    tn = int((~y & ~alert).sum())
    missed = float(amount[y & ~alert].sum())
    review = (tp + fp) * cfg.review_cost
    friction = fp * cfg.friction_cost
    total = missed + review + friction
    no_model = float(amount[y].sum())
    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "missed_fraud_value": missed,
        "review_cost": review,
        "friction_cost": friction,
        "total_cost": total,
        "no_model_cost": no_model,
        "net_savings": no_model - total,
    }
