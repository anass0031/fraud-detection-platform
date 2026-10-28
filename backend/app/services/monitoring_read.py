"""Read side of monitoring: shapes the daily tables (and a few live queries) for the API."""

from __future__ import annotations

import datetime as dt
from collections import defaultdict
from collections.abc import Sequence

import numpy as np
from fraudml.evaluate.metrics import pr_auc
from fraudml.monitoring.drift import PSI_MODERATE, PSI_SIGNIFICANT, SCORE
from sqlalchemy import Date, cast, func, select
from sqlalchemy.orm import Session

from app.db.models import (
    Case,
    Decision,
    DriftReport,
    MetricSnapshot,
    ModelVersion,
    Prediction,
    Transaction,
)
from app.services.settings import COSTS, get_setting

OPEN_STATUSES = ("new", "assigned", "escalated")


def active_model(db: Session) -> ModelVersion | None:
    return db.scalar(select(ModelVersion).where(ModelVersion.status == "active"))


def _between(column, start: dt.date | None, end: dt.date | None):
    conds = []
    if start is not None:
        conds.append(column >= start)
    if end is not None:
        conds.append(column <= end)
    return conds


def _safe(num: float, den: float) -> float | None:
    return num / den if den else None


# ---------------------------------------------------------------- performance


def snapshots(
    db: Session, model_id: int, start: dt.date | None, end: dt.date | None
) -> list[MetricSnapshot]:
    return list(
        db.scalars(
            select(MetricSnapshot)
            .where(
                MetricSnapshot.model_version_id == model_id,
                *_between(MetricSnapshot.day, start, end),
            )
            .order_by(MetricSnapshot.day)
        )
    )


def row(s: MetricSnapshot) -> dict:
    return {
        "day": s.day,
        "n": s.n,
        "n_fraud": s.n_fraud,
        "alerts": s.alerts,
        "tp": s.tp,
        "fp": s.fp,
        "fn": s.fn,
        "precision": s.precision,
        "recall": s.recall,
        "pr_auc": s.pr_auc,
        "precision_at_capacity": s.precision_at_capacity,
        "fraud_value_caught": _safe(s.fraud_value_caught, s.fraud_value),
        "total_cost": s.total_cost,
        "no_model_cost": s.no_model_cost,
        "net_savings": s.net_savings,
    }


def period_pr_auc(db: Session, model_id: int, first: dt.date, last: dt.date) -> float | None:
    """PR-AUC cannot be summed over days: recompute it on the period's labelled rows."""
    start = dt.datetime.combine(first, dt.time())
    end = dt.datetime.combine(last + dt.timedelta(days=1), dt.time())
    rows = db.execute(
        select(Transaction.ground_truth, Prediction.fraud_score)
        .join(Prediction, Prediction.transaction_id == Transaction.id)
        .where(
            Prediction.model_version_id == model_id,
            Transaction.transaction_time >= start,
            Transaction.transaction_time < end,
            Transaction.ground_truth_revealed_at.is_not(None),
            Transaction.ground_truth.is_not(None),
        )
    ).all()
    if not rows:
        return None
    y = np.array([r[0] for r in rows], dtype=bool)
    if y.all() or not y.any():
        return None
    return pr_auc(y, np.array([r[1] for r in rows], dtype="float64"))


def totals(db: Session, model_id: int, snaps: Sequence[MetricSnapshot]) -> dict | None:
    """Exact sums over days (counts and dollars), PR-AUC recomputed."""
    if not snaps:
        return None
    s = {k: sum(getattr(x, k) for x in snaps) for k in (
        "n", "n_fraud", "alerts", "tp", "fraud_value", "fraud_value_caught",
        "total_cost", "no_model_cost", "net_savings",
    )}  # fmt: skip
    first, last = snaps[0].day, snaps[-1].day
    return {
        "first_day": first,
        "last_day": last,
        "days": len(snaps),
        "n": s["n"],
        "n_fraud": s["n_fraud"],
        "alerts": s["alerts"],
        "alerts_per_day": s["alerts"] / len(snaps),
        "precision": _safe(s["tp"], s["alerts"]),
        "recall": _safe(s["tp"], s["n_fraud"]),
        "pr_auc": period_pr_auc(db, model_id, first, last),
        "fraud_value_caught": _safe(s["fraud_value_caught"], s["fraud_value"]),
        "total_cost": s["total_cost"],
        "no_model_cost": s["no_model_cost"],
        "net_savings": s["net_savings"],
    }


# ---------------------------------------------------------------- drift


def drift(db: Session, model_id: int, start: dt.date | None, end: dt.date | None) -> dict:
    reports = db.scalars(
        select(DriftReport)
        .where(DriftReport.model_version_id == model_id, *_between(DriftReport.day, start, end))
        .order_by(DriftReport.day)
    ).all()
    days = sorted({r.day for r in reports})
    by_feature: dict[str, list[DriftReport]] = defaultdict(list)
    for r in reports:
        by_feature[r.feature_name].append(r)

    score = [
        {"day": r.day, "psi": r.psi, "ks_stat": r.ks_stat, "severity": r.severity}
        for r in by_feature.pop(SCORE, [])
    ]
    features = []
    for name, rs in by_feature.items():
        psis = [r.psi for r in rs]
        features.append(
            {
                "feature": name,
                "max_psi": max(psis),
                "mean_psi": sum(psis) / len(psis),
                "latest_psi": rs[-1].psi,
                "latest_severity": rs[-1].severity,
                "days_significant": sum(r.severity == "significant" for r in rs),
                "cells": [{"day": r.day, "psi": r.psi, "severity": r.severity} for r in rs],
            }
        )
    features.sort(key=lambda f: f["max_psi"], reverse=True)
    return {
        "days": days,
        "features": features,
        "score": score,
        "thresholds": {"moderate": PSI_MODERATE, "significant": PSI_SIGNIFICANT},
    }


# ---------------------------------------------------------------- alerts and backlog


def alerts(db: Session, start: dt.date | None, end: dt.date | None) -> dict:
    day = cast(Transaction.transaction_time, Date)
    is_open = Case.status.in_(OPEN_STATUSES)
    rows = db.execute(
        select(
            day.label("day"),
            func.count().filter(Case.reason == "model_score"),
            func.count().filter(Case.reason == "anomaly"),
            func.count().filter(is_open),
            func.count().filter(~is_open),
        )
        .join(Transaction, Case.transaction_id == Transaction.id)
        .where(*_between(day, start, end))
        .group_by(day)
        .order_by(day)
    ).all()
    backlog = dict(
        db.execute(select(Case.status, func.count()).where(is_open).group_by(Case.status)).all()
    )
    costs, _ = get_setting(db, COSTS)
    return {
        "days": [
            {"day": d, "model_alerts": a, "unusual": u, "open": o, "closed": c}
            for d, a, u, o, c in rows
        ],
        "backlog": {s: int(backlog.get(s, 0)) for s in OPEN_STATUSES},
        "capacity_per_day": int(costs["capacity_per_day"]),
    }


# ---------------------------------------------------------------- analyst feedback


def feedback(db: Session) -> dict:
    by_type = dict(
        db.execute(select(Decision.decision, func.count()).group_by(Decision.decision)).all()
    )
    # Final decision of each closed case = its latest non-escalate decision.
    latest = (
        select(Decision.case_id, func.max(Decision.id).label("decision_id"))
        .where(Decision.decision != "escalate")
        .group_by(Decision.case_id)
        .subquery()
    )
    closed = db.execute(
        select(
            Decision.decision,
            Case.created_at,
            Case.closed_at,
            Transaction.ground_truth,
            Transaction.ground_truth_revealed_at,
        )
        .join(latest, latest.c.decision_id == Decision.id)
        .join(Case, Case.id == Decision.case_id)
        .join(Transaction, Transaction.id == Case.transaction_id)
        .where(Case.status == "closed")
    ).all()
    confirmed = sum(d == "confirmed_fraud" for d, *_ in closed)
    checked = [(d, gt) for d, _c, _cl, gt, rev in closed if rev is not None and gt is not None]
    matching = sum((d == "confirmed_fraud") == bool(gt) for d, gt in checked)
    hours = [(cl - c).total_seconds() / 3600 for _d, c, cl, _gt, _rev in closed if cl and c]
    return {
        "decisions": {
            k: int(by_type.get(k, 0)) for k in ("confirmed_fraud", "false_positive", "escalate")
        },
        "cases_closed": len(closed),
        "confirmed_share": _safe(confirmed, len(closed)),
        "checked_against_labels": len(checked),
        "matching_labels": matching,
        "mean_hours_to_decision": sum(hours) / len(hours) if hours else None,
    }
