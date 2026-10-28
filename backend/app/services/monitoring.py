"""Monitoring of one day: performance on revealed labels + drift against the training reference.

Uses the same code as the offline evaluation (`fraudml.evaluate`) on the stored predictions:
`is_alert` as stored (a later threshold change does not rewrite history), costs and capacity
from the `costs` setting. Results are upserted, so a day can be recomputed at any time.
"""

from __future__ import annotations

import datetime as dt
import json
from functools import cache
from pathlib import Path
from typing import Protocol

import numpy as np
import pandas as pd
from fraudml import config
from fraudml.evaluate.cost import CostConfig, cost_breakdown
from fraudml.evaluate.metrics import pr_auc, precision_at_capacity
from fraudml.monitoring.drift import SCORE, FeatureReference, load_reference, measure
from sqlalchemy import select
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.orm import Session

from app.db.models import DriftReport, MetricSnapshot, Prediction, Transaction
from app.services.settings import COSTS, get_setting

MONITORING_DIR = config.ARTIFACTS_DIR / "monitoring"

# A one-day window always holds a single weekday, so its PSI against the 7-day training mix is
# huge every day (≈ 7) without meaning anything. Calendar features are not monitored daily.
NOT_MONITORED_DAILY = frozenset({"weekday"})


class ModelRef(Protocol):
    """What monitoring needs to know about a model (a ScoringBundle fits)."""

    model_version_id: int
    version: str


def reference_path(version: str) -> Path:
    return MONITORING_DIR / f"drift_reference-{version}.json"


@cache
def load_drift_reference(version: str) -> dict[str, FeatureReference] | None:
    path = reference_path(version)
    if not path.exists():
        return None
    return load_reference(json.loads(path.read_text(encoding="utf-8")))


def day_frame(db: Session, model_version_id: int, day: dt.date) -> pd.DataFrame:
    """The day's scored transactions for this model (one row each, with label if revealed)."""
    start = dt.datetime.combine(day, dt.time())
    rows = db.execute(
        select(
            Transaction.amount,
            Transaction.ground_truth,
            Transaction.ground_truth_revealed_at,
            Transaction.features,
            Prediction.fraud_score,
            Prediction.expected_loss,
            Prediction.is_alert,
            Prediction.is_unusual,
        )
        .join(Prediction, Prediction.transaction_id == Transaction.id)
        .where(
            Prediction.model_version_id == model_version_id,
            Transaction.transaction_time >= start,
            Transaction.transaction_time < start + dt.timedelta(days=1),
        )
        .order_by(Transaction.transaction_dt, Transaction.external_id)
    ).all()
    return pd.DataFrame(rows, columns=[
        "amount", "ground_truth", "revealed_at", "features",
        "fraud_score", "expected_loss", "is_alert", "is_unusual",
    ])  # fmt: skip


def cost_config(db: Session) -> CostConfig:
    value, _ = get_setting(db, COSTS)
    return CostConfig(
        review_cost=float(value["review_cost"]),
        friction_cost=float(value["friction_cost"]),
        capacity_per_day=int(value["capacity_per_day"]),
    )


def day_metrics(frame: pd.DataFrame, cfg: CostConfig) -> dict | None:
    """Performance of one day on its revealed labels (None if no label is revealed yet)."""
    labelled = frame[frame["revealed_at"].notna() & frame["ground_truth"].notna()]
    if labelled.empty:
        return None
    y = labelled["ground_truth"].astype(bool).to_numpy()
    amount = labelled["amount"].astype("float64").to_numpy()
    alert = labelled["is_alert"].astype(bool).to_numpy()
    cost = cost_breakdown(y, amount, alert, cfg)
    n_fraud = int(y.sum())
    alerts = int(alert.sum())
    same_day = np.zeros(len(labelled))
    return {
        "n": len(labelled),
        "n_fraud": n_fraud,
        "alerts": alerts,
        "tp": cost["tp"],
        "fp": cost["fp"],
        "fn": cost["fn"],
        "unusual_flags": int(labelled["is_unusual"].astype(bool).sum()),
        "precision": cost["tp"] / alerts if alerts else None,
        "recall": cost["tp"] / n_fraud if n_fraud else None,
        "pr_auc": pr_auc(y, labelled["fraud_score"].to_numpy()) if 0 < n_fraud < len(y) else None,
        # The day's top-k by expected loss (what analysts would review with k = capacity).
        "precision_at_capacity": precision_at_capacity(
            y, labelled["expected_loss"].to_numpy(), same_day, cfg.capacity_per_day
        ),
        "fraud_value": float(amount[y].sum()),
        "fraud_value_caught": float(amount[y & alert].sum()),
        "total_cost": cost["total_cost"],
        "no_model_cost": cost["no_model_cost"],
        "net_savings": cost["net_savings"],
    }


def day_drift(frame: pd.DataFrame, refs: dict[str, FeatureReference]) -> list[dict]:
    """PSI/KS of every reference feature and of the score for one day."""
    if frame.empty:
        return []
    features = pd.DataFrame(list(frame["features"]))
    out = []
    for name, ref in refs.items():
        if name in NOT_MONITORED_DAILY:
            continue
        values = frame["fraud_score"] if name == SCORE else features.get(name)
        if values is None:
            continue
        r = measure(ref, values.to_numpy() if ref.kind == "numeric" else values)
        out.append(
            {
                "feature_name": name,
                "n": r.n,
                "psi": r.psi,
                "ks_stat": r.ks_stat,
                "ks_pvalue": r.ks_pvalue,
                "severity": r.severity,
            }
        )
    return out


def compute_day(db: Session, model: ModelRef, day: dt.date) -> tuple[bool, int]:
    """Upsert the day's metric snapshot and drift rows. Returns (metrics stored, drift rows)."""
    frame = day_frame(db, model.model_version_id, day)
    metrics = day_metrics(frame, cost_config(db))
    if metrics is not None:
        values = {"model_version_id": model.model_version_id, "day": day, **metrics}
        stmt = insert(MetricSnapshot).values(values)
        db.execute(
            stmt.on_conflict_do_update(
                constraint="uq_metric_snapshots_model_day",
                set_={k: stmt.excluded[k] for k in metrics}
                | {"created_at": dt.datetime.now(dt.UTC)},
            )
        )

    refs = load_drift_reference(model.version)
    rows = day_drift(frame, refs) if refs else []
    for r in rows:
        stmt = insert(DriftReport).values(
            {"model_version_id": model.model_version_id, "day": day, **r}
        )
        db.execute(
            stmt.on_conflict_do_update(
                constraint="uq_drift_reports_model_day_feature",
                set_={k: stmt.excluded[k] for k in r if k != "feature_name"}
                | {"created_at": dt.datetime.now(dt.UTC)},
            )
        )
    db.flush()
    return metrics is not None, len(rows)
