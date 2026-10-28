import datetime as dt

import pytest
from sqlalchemy.orm import Session

from app.db.models import Decision, DriftReport, MetricSnapshot
from app.services.cases import open_case
from tests.conftest import auth_header

URL = "/api/v1/monitoring"
D1, D2 = dt.date(2018, 4, 28), dt.date(2018, 4, 29)
NOW = dt.datetime.now(dt.UTC)


def snapshot(model_id: int, day: dt.date, **kw) -> MetricSnapshot:
    base = dict(
        model_version_id=model_id, day=day, n=1000, n_fraud=30, alerts=150, tp=40, fp=110,
        fn=10, unusual_flags=2, precision=40 / 150, recall=0.8, pr_auc=0.5,
        precision_at_capacity=0.27, fraud_value=10_000.0, fraud_value_caught=6_000.0,
        total_cost=5_000.0, no_model_cost=10_000.0, net_savings=5_000.0,
    )  # fmt: skip
    return MetricSnapshot(**{**base, **kw})


@pytest.fixture
def headers(api, make_user):
    return auth_header(api, make_user("analyst"))


@pytest.fixture
def model_id(make_prediction) -> int:
    """Registers the active test model (via a first prediction) and returns its id."""
    return make_prediction(when=dt.datetime(2018, 4, 1)).model_version_id


def test_requires_login(api):
    for path in ("performance", "drift", "alerts", "feedback"):
        assert api.get(f"{URL}/{path}").status_code == 401


def test_empty_database(api, headers):
    perf = api.get(f"{URL}/performance", headers=headers).json()
    assert perf == {"model_version": None, "days": [], "total": None, "last_7_days": None}
    drift = api.get(f"{URL}/drift", headers=headers).json()
    assert (drift["days"], drift["features"], drift["score"]) == ([], [], [])
    alerts = api.get(f"{URL}/alerts", headers=headers).json()
    assert alerts["days"] == [] and alerts["backlog"] == {"new": 0, "assigned": 0, "escalated": 0}
    assert alerts["capacity_per_day"] == 150


def test_performance_sums_days_exactly(
    api, db_session: Session, headers, make_prediction, model_id
):
    db_session.add_all(
        [
            snapshot(model_id, D1),
            snapshot(model_id, D2, alerts=100, tp=50, fp=50, n_fraud=70, net_savings=-500.0,
                     fraud_value=20_000.0, fraud_value_caught=4_000.0),
        ]
    )  # fmt: skip
    # Labelled predictions for the recomputed PR-AUC: fraud ranked above legit -> 1.0
    for score, fraud in [(0.9, True), (0.8, True), (0.2, False), (0.1, False)]:
        p = make_prediction(score=score, when=dt.datetime(2018, 4, 28, 12), ground_truth=fraud)
        p.transaction.ground_truth_revealed_at = NOW
    hidden = make_prediction(score=0.01, when=dt.datetime(2018, 4, 28, 13), ground_truth=True)
    assert hidden.transaction.ground_truth_revealed_at is None  # not revealed: ignored
    db_session.flush()

    body = api.get(f"{URL}/performance", headers=headers).json()
    assert body["model_version"] == "test-model"
    assert [d["day"] for d in body["days"]] == ["2018-04-28", "2018-04-29"]
    assert body["days"][0]["fraud_value_caught"] == pytest.approx(0.6)
    t = body["total"]
    assert (t["days"], t["n"], t["alerts"], t["n_fraud"]) == (2, 2000, 250, 100)
    assert t["alerts_per_day"] == 125
    assert t["precision"] == pytest.approx(90 / 250)  # (40 + 50) / (150 + 100), not a mean
    assert t["recall"] == pytest.approx(0.9)
    assert t["fraud_value_caught"] == pytest.approx(10_000 / 30_000)
    assert t["net_savings"] == 4_500
    assert t["pr_auc"] == pytest.approx(1.0)
    assert body["last_7_days"] == t  # fewer than 7 days: same period

    one = api.get(f"{URL}/performance", headers=headers, params={"from": "2018-04-29"}).json()
    assert [d["day"] for d in one["days"]] == ["2018-04-29"]
    bad = api.get(f"{URL}/performance", headers=headers, params={"from": D2, "to": D1})
    assert bad.status_code == 422


def test_drift_sorted_by_max_psi_with_score_apart(api, db_session: Session, headers, model_id):
    def rep(day, feature, psi, severity):
        return DriftReport(
            model_version_id=model_id, day=day, feature_name=feature, n=100, psi=psi,
            ks_stat=0.1, ks_pvalue=0.5, severity=severity,
        )  # fmt: skip

    db_session.add_all(
        [
            rep(D1, "amt_log", 0.02, "stable"),
            rep(D2, "amt_log", 0.05, "stable"),
            rep(D1, "D15n", 0.3, "significant"),
            rep(D2, "D15n", 1.2, "significant"),
            rep(D1, "__score__", 0.12, "moderate"),
            rep(D2, "__score__", 0.08, "stable"),
        ]
    )
    db_session.flush()
    body = api.get(f"{URL}/drift", headers=headers).json()
    assert body["days"] == ["2018-04-28", "2018-04-29"]
    assert [f["feature"] for f in body["features"]] == ["D15n", "amt_log"]
    top = body["features"][0]
    assert (top["max_psi"], top["latest_psi"], top["days_significant"]) == (1.2, 1.2, 2)
    assert top["mean_psi"] == pytest.approx(0.75)
    assert [c["psi"] for c in top["cells"]] == [0.3, 1.2]
    assert [s["severity"] for s in body["score"]] == ["moderate", "stable"]
    assert body["thresholds"] == {"moderate": 0.1, "significant": 0.25}


def test_alerts_per_day_and_backlog(api, db_session: Session, headers, make_prediction):
    c1 = open_case(db_session, make_prediction(when=dt.datetime(2018, 4, 28, 9)))
    open_case(
        db_session, make_prediction(alert=False, unusual=True, when=dt.datetime(2018, 4, 28, 10))
    )
    c3 = open_case(db_session, make_prediction(when=dt.datetime(2018, 4, 29, 9)))
    c1.status = "closed"
    c3.status = "escalated"
    db_session.flush()
    body = api.get(f"{URL}/alerts", headers=headers).json()
    assert body["days"] == [
        {"day": "2018-04-28", "model_alerts": 1, "unusual": 1, "open": 1, "closed": 1},
        {"day": "2018-04-29", "model_alerts": 1, "unusual": 0, "open": 1, "closed": 0},
    ]
    assert body["backlog"] == {"new": 1, "assigned": 0, "escalated": 1}


def test_feedback_counts_and_agreement(
    api, db_session: Session, headers, make_prediction, make_user
):
    analyst = make_user("analyst")

    def closed_case(decisions: list[str], fraud: bool, revealed: bool):
        p = make_prediction(ground_truth=fraud)
        if revealed:
            p.transaction.ground_truth_revealed_at = NOW
        c = open_case(db_session, p)
        c.status, c.closed_at = "closed", c.created_at + dt.timedelta(hours=2)
        for d in decisions:
            db_session.add(Decision(case_id=c.id, analyst_id=analyst.id, decision=d))
        db_session.flush()

    closed_case(["confirmed_fraud"], fraud=True, revealed=True)  # agrees with the label
    closed_case(["escalate", "false_positive"], fraud=True, revealed=True)  # disagrees
    closed_case(["false_positive"], fraud=False, revealed=False)  # label still hidden
    body = api.get(f"{URL}/feedback", headers=headers).json()
    assert body["decisions"] == {"confirmed_fraud": 1, "false_positive": 2, "escalate": 1}
    assert body["cases_closed"] == 3
    assert body["confirmed_share"] == pytest.approx(1 / 3)
    assert (body["checked_against_labels"], body["matching_labels"]) == (2, 1)
    assert body["mean_hours_to_decision"] == pytest.approx(2.0, abs=0.01)


def test_no_label_field_in_any_response(api, headers, make_prediction):
    make_prediction(ground_truth=True)
    for path in ("performance", "drift", "alerts", "feedback"):
        text = api.get(f"{URL}/{path}", headers=headers).text
        assert "ground_truth" not in text and "isFraud" not in text
