import datetime as dt

import pandas as pd
import pytest
from fraudml.evaluate.cost import CostConfig
from fraudml.monitoring.drift import SCORE, build_feature_reference
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.db.models import DriftReport, MetricSnapshot
from app.services import monitoring
from app.services.monitoring import compute_day, day_drift, day_metrics

DAY = dt.date(2018, 4, 28)
REVEALED = pd.Timestamp("2026-10-28", tz="UTC")


def frame(rows: list[tuple]) -> pd.DataFrame:
    """rows: (amount, is_fraud, alert, score, expected_loss, revealed)."""
    return pd.DataFrame(
        [
            {
                "amount": a,
                "ground_truth": y,
                "revealed_at": REVEALED if rev else None,
                "features": {"amt_log": a / 100, "card6": "credit" if y else "debit"},
                "fraud_score": s,
                "expected_loss": el,
                "is_alert": al,
                "is_unusual": False,
            }
            for a, y, al, s, el, rev in rows
        ]
    )


ROWS = [
    # amount, fraud, alert, score, expected loss, revealed
    (100.0, True, True, 0.6, 60.0, True),  # TP
    (50.0, True, False, 0.1, 5.0, True),  # FN (missed $50)
    (200.0, False, True, 0.3, 60.0, True),  # FP
    (10.0, False, False, 0.01, 0.1, True),  # TN
    (999.0, True, True, 0.9, 899.0, False),  # label not revealed yet: ignored
]


def test_day_metrics_match_the_cost_model():
    m = day_metrics(frame(ROWS), CostConfig(review_cost=5, friction_cost=2, capacity_per_day=1))
    assert m is not None
    assert (m["n"], m["n_fraud"], m["alerts"], m["tp"], m["fp"], m["fn"]) == (4, 2, 2, 1, 1, 1)
    assert m["precision"] == 0.5
    assert m["recall"] == 0.5
    assert m["fraud_value"] == 150.0
    assert m["fraud_value_caught"] == 100.0
    # missed 50 + 2 alerts x $5 + 1 false alarm x $2
    assert m["total_cost"] == 62.0
    assert m["no_model_cost"] == 150.0
    assert m["net_savings"] == 88.0
    # capacity 1: the single top expected loss is a tie (60, 60) -> first one = the fraud
    assert m["precision_at_capacity"] == 1.0
    # ranking by score: fraud 0.6, legit 0.3, fraud 0.1, legit 0.01 -> AP = (1/1 + 2/3) / 2
    assert m["pr_auc"] == pytest.approx(5 / 6)


def test_no_revealed_label_no_metrics():
    rows = [(r[0], r[1], r[2], r[3], r[4], False) for r in ROWS]
    assert day_metrics(frame(rows), CostConfig()) is None


def test_day_drift_covers_features_and_score():
    reference = frame(ROWS)
    refs = {
        "amt_log": build_feature_reference(
            "amt_log", pd.Series([r["amt_log"] for r in reference["features"]]), "numeric"
        ),
        "card6": build_feature_reference(
            "card6", pd.Series(["credit", "debit"] * 10), "categorical"
        ),
        SCORE: build_feature_reference(SCORE, reference["fraud_score"], "numeric"),
        "not_in_data": build_feature_reference("not_in_data", pd.Series([1.0, 2.0]), "numeric"),
        "weekday": build_feature_reference("weekday", pd.Series(range(7)), "numeric"),
    }
    rows = {r["feature_name"]: r for r in day_drift(reference, refs)}
    assert set(rows) == {"amt_log", "card6", SCORE}  # weekday: calendar, not monitored daily
    assert rows["amt_log"]["psi"] == pytest.approx(0, abs=1e-9)
    assert rows["card6"]["ks_stat"] is None
    assert all(r["n"] == 5 for r in rows.values())


def test_compute_day_upserts(db_session: Session, make_prediction, monkeypatch):
    """Real tables: a re-run updates the same rows instead of adding new ones."""
    when = dt.datetime(2018, 4, 28, 10)
    p1 = make_prediction(amount=100.0, score=0.6, alert=True, when=when, ground_truth=True)
    make_prediction(amount=40.0, score=0.05, alert=False, when=when, ground_truth=False)
    for p in db_session.scalars(select(MetricSnapshot)):  # none yet
        raise AssertionError(p)
    p1.transaction.ground_truth_revealed_at = dt.datetime.now(dt.UTC)
    db_session.flush()

    refs = {SCORE: build_feature_reference(SCORE, pd.Series([0.1, 0.2, 0.6]), "numeric")}
    monkeypatch.setattr(monitoring, "load_drift_reference", lambda version: refs)

    class Model:
        model_version_id = p1.model_version_id
        version = "test-model"

    assert compute_day(db_session, Model, DAY) == (True, 1)
    assert compute_day(db_session, Model, DAY) == (True, 1)
    snap = db_session.scalar(select(MetricSnapshot))
    assert (snap.n, snap.tp, snap.alerts) == (1, 1, 1)  # only the revealed row counts
    assert db_session.scalar(select(func.count()).select_from(MetricSnapshot)) == 1
    drift = db_session.scalar(select(DriftReport))
    assert (drift.feature_name, drift.n) == (SCORE, 2)  # drift uses every scored row
    assert db_session.scalar(select(func.count()).select_from(DriftReport)) == 1
