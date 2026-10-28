import pytest
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.db.models import AuditLog, Case, Prediction
from app.services import settings as svc
from tests.conftest import auth_header
from tests.test_score import PAYLOAD, client, needs_artifacts  # noqa: F401 (fixture)

URL = "/api/v1/settings"
THRESHOLDS = {"alert_threshold": 30.0, "risk_bands": [0.1, 0.3, 0.6]}
COSTS = {"review_cost": 6.5, "friction_cost": 1.0, "capacity_per_day": 200}


@pytest.fixture
def admin(make_user):
    return make_user("admin")


@pytest.fixture
def admin_h(api, admin):
    return auth_header(api, admin)


def test_defaults_without_stored_settings(api, admin_h, make_prediction):
    make_prediction()  # registers an active test model (threshold 20)
    body = api.get(f"{URL}/thresholds", headers=admin_h).json()
    assert body["alert_threshold"] is None
    assert body["risk_bands"] == [0.05, 0.2, 0.5]
    assert body["effective_threshold"] == 20.0
    assert (body["model_version"], body["model_threshold"]) == ("test-model", 20.0)
    assert body["alert_policy"] == "expected_loss"
    assert body["updated_by"] is None

    costs = api.get(f"{URL}/costs", headers=admin_h).json()
    assert (costs["review_cost"], costs["friction_cost"], costs["capacity_per_day"]) == (
        5.0,
        2.0,
        150,
    )


def test_update_thresholds_is_saved_and_audited(api, db_session, admin, admin_h):
    resp = api.put(f"{URL}/thresholds", headers=admin_h, json=THRESHOLDS)
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["alert_threshold"] == 30.0
    assert body["effective_threshold"] == 30.0
    assert body["updated_by"]["id"] == admin.id

    # Back to the model's own threshold.
    api.put(f"{URL}/thresholds", headers=admin_h, json={**THRESHOLDS, "alert_threshold": None})
    assert api.get(f"{URL}/thresholds", headers=admin_h).json()["alert_threshold"] is None

    first, second = db_session.scalars(
        select(AuditLog).where(AuditLog.action == "settings.update").order_by(AuditLog.id)
    ).all()
    assert (first.entity_type, first.entity_id, first.actor_id) == (
        "setting",
        "thresholds",
        admin.id,
    )
    assert first.before == {"alert_threshold": None, "risk_bands": [0.05, 0.2, 0.5]}
    assert first.after == THRESHOLDS
    assert second.before == THRESHOLDS


def test_update_costs(api, db_session, admin_h):
    resp = api.put(f"{URL}/costs", headers=admin_h, json=COSTS)
    assert resp.status_code == 200, resp.text
    assert resp.json()["capacity_per_day"] == 200
    assert svc.get_setting(db_session, svc.COSTS)[0] == COSTS


@pytest.mark.parametrize(
    "body",
    [
        {**THRESHOLDS, "alert_threshold": 0},
        {**THRESHOLDS, "alert_threshold": -5},
        {**THRESHOLDS, "risk_bands": [0.1, 0.3]},
        {**THRESHOLDS, "risk_bands": [0.3, 0.1, 0.6]},
        {**THRESHOLDS, "risk_bands": [0.1, 0.1, 0.6]},
        {**THRESHOLDS, "risk_bands": [0.0, 0.3, 0.6]},
        {**THRESHOLDS, "risk_bands": [0.1, 0.3, 1.0]},
        {"risk_bands": [0.1, 0.3, 0.6]},  # alert_threshold must be given (null is explicit)
        {**THRESHOLDS, "extra": 1},
    ],
)
def test_invalid_thresholds(api, admin_h, body):
    assert api.put(f"{URL}/thresholds", headers=admin_h, json=body).status_code == 422


@pytest.mark.parametrize(
    "body",
    [
        {**COSTS, "review_cost": -1},
        {**COSTS, "friction_cost": -0.5},
        {**COSTS, "capacity_per_day": 0},
        {**COSTS, "capacity_per_day": 10_001},
        {"review_cost": 5, "friction_cost": 2},
    ],
)
def test_invalid_costs(api, admin_h, body):
    assert api.put(f"{URL}/costs", headers=admin_h, json=body).status_code == 422


def test_settings_are_admin_only(api, make_user):
    h = auth_header(api, make_user("analyst"))
    for path in ("thresholds", "costs"):
        assert api.get(f"{URL}/{path}", headers=h).status_code == 403
    assert api.put(f"{URL}/thresholds", headers=h, json=THRESHOLDS).status_code == 403
    assert api.put(f"{URL}/costs", headers=h, json=COSTS).status_code == 403
    assert api.get(f"{URL}/costs").status_code == 401


def test_alert_policy_falls_back_to_the_model(db_session: Session, admin):
    assert svc.alert_policy(db_session, 21.8) == svc.AlertPolicy(21.8)
    svc.update_setting(db_session, svc.THRESHOLDS, THRESHOLDS, admin)
    policy = svc.alert_policy(db_session, 21.8)
    assert (policy.threshold, policy.risk_bands, policy.overridden) == (30.0, (0.1, 0.3, 0.6), True)


@needs_artifacts
def test_threshold_change_applies_to_the_next_score_only(client, db_session, admin):  # noqa: F811
    first = client.post("/api/v1/score", json=PAYLOAD).json()
    assert first["is_alert"] is False  # expected loss far below the model's $21.82
    assert first["expected_loss"] < 21.8

    # Admin lowers the threshold to 1 cent and narrows the bands.
    low = 0.01
    svc.update_setting(
        db_session,
        svc.THRESHOLDS,
        {"alert_threshold": low, "risk_bands": [0.001, 0.002, 0.003]},
        admin,
    )
    second = client.post(
        "/api/v1/score", json={**PAYLOAD, "TransactionID": PAYLOAD["TransactionID"] + 10**7}
    ).json()
    assert second["is_alert"] is True
    assert second["alert_threshold"] == pytest.approx(low)
    assert second["risk_level"] == "critical"

    # The first prediction is unchanged and only the second transaction opened a case.
    preds = db_session.scalars(select(Prediction).order_by(Prediction.id)).all()
    assert [p.is_alert for p in preds] == [False, True]
    cases = db_session.scalars(select(Case)).all()
    assert [c.prediction_id for c in cases] == [preds[1].id]
