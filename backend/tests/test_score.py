from collections.abc import Generator

import pytest
from fastapi.testclient import TestClient
from fraudml.registry import MODELS_DIR
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.db.models import Prediction, Transaction
from app.db.session import get_db
from app.main import app
from app.services.model_store import ModelStore, get_model_store, register
from app.services.scoring import risk_level
from tests.conftest import auth_header

MODEL = "lgbm-2026.10.27-1"
ANOMALY = "iforest-2026.10.27-1"

# A real test-period transaction (IEEE-CIS TransactionID 3479348), used as a template.
PAYLOAD = {
    "TransactionID": 3479348,
    "TransactionDT": 12851215,
    "TransactionAmt": 87.0,
    "ProductCD": "W",
    "card1": 15651,
    "card2": 417.0,
    "card4": "visa",
    "card6": "debit",
    "addr1": 330.0,
    "dist1": 3.0,
    "P_emaildomain": "gmail.com",
    "R_emaildomain": None,
    "DeviceType": None,
    "DeviceInfo": None,
    "id_01": None,
    "id_30": None,
    "id_31": None,
    "D1": 56.0,
    "D4": 56.0,
    "D10": 145.0,
    "D15": 56.0,
    **{f"C{i}": v for i, v in enumerate([2, 1, 0, 0, 0, 1, 0, 0, 1, 0, 1, 1, 7, 2], start=1)},
    **dict(zip([f"M{i}" for i in range(1, 10)], "T T T M0 F F F T T".split(), strict=True)),
}

needs_artifacts = pytest.mark.skipif(
    not (MODELS_DIR / MODEL).exists() or not (MODELS_DIR / ANOMALY).exists(),
    reason="trained artifacts not available",
)


@pytest.fixture
def client(db_session: Session, make_user) -> Generator[TestClient, None, None]:
    """Logged-in analyst client on the test database with the real model loaded."""
    register(db_session, MODEL, ANOMALY, activate=True)
    store = ModelStore()
    assert store.load_active(db_session) is not None
    app.dependency_overrides[get_db] = lambda: db_session
    app.dependency_overrides[get_model_store] = lambda: store
    try:
        c = TestClient(app)
        c.headers.update(auth_header(c, make_user("analyst")))
        yield c
    finally:
        app.dependency_overrides.clear()


def _payload(**changes) -> dict:
    return {**PAYLOAD, **changes}


def _count(db: Session, model) -> int:
    return db.scalar(select(func.count()).select_from(model))


@needs_artifacts
def test_score_returns_explained_prediction_and_stores_it(client, db_session):
    resp = client.post("/api/v1/score", json=PAYLOAD)
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["transaction_id"] == PAYLOAD["TransactionID"]
    assert body["model_version"] == MODEL and body["anomaly_version"] == ANOMALY
    assert 0.0 <= body["fraud_score"] <= 1.0
    assert body["expected_loss"] == pytest.approx(body["fraud_score"] * 87.0)
    assert body["is_alert"] == (body["expected_loss"] >= body["alert_threshold"])
    assert body["risk_level"] == risk_level(body["fraud_score"])
    assert len(body["top_factors"]) == 10
    assert all(f["text"] for f in body["top_factors"])
    assert body["already_scored"] is False

    txn = db_session.scalar(select(Transaction).where(Transaction.external_id == 3479348))
    assert txn.source == "realtime"
    assert txn.ground_truth is None
    assert txn.features["uid_n_prior"] == 0.0  # empty test DB: no history
    pred = db_session.scalar(select(Prediction).where(Prediction.transaction_id == txn.id))
    assert pred.fraud_score == body["fraud_score"]
    assert len(pred.shap_values["factors"]) == 10


@needs_artifacts
def test_second_transaction_of_a_customer_sees_the_first(client, db_session):
    first = _payload(TransactionID=1, TransactionDT=12_000_000)
    second = _payload(TransactionID=2, TransactionDT=12_003_600, P_emaildomain="yahoo.com")
    assert client.post("/api/v1/score", json=first).status_code == 200
    assert client.post("/api/v1/score", json=second).status_code == 200
    rows = {
        t.external_id: t.features
        for t in db_session.scalars(select(Transaction).where(Transaction.external_id < 3))
    }
    assert rows[1]["uid_n_prior"] == 0.0
    assert rows[2]["uid_n_prior"] == 1.0
    assert rows[2]["secs_since_prev_uid"] == 3600.0
    assert rows[2]["uid_new_p_email"] == 1.0


@needs_artifacts
def test_same_transaction_twice_returns_the_stored_prediction(client, db_session):
    a = client.post("/api/v1/score", json=PAYLOAD).json()
    b = client.post("/api/v1/score", json=PAYLOAD).json()
    assert b["already_scored"] is True
    assert b["fraud_score"] == a["fraud_score"]
    assert _count(db_session, Prediction) == 1


@needs_artifacts
def test_same_id_with_other_data_is_a_conflict(client):
    assert client.post("/api/v1/score", json=PAYLOAD).status_code == 200
    resp = client.post("/api/v1/score", json=_payload(TransactionAmt=999.0))
    assert resp.status_code == 409


@needs_artifacts
def test_unknown_categories_and_missing_fields_are_scored(client):
    minimal = {"TransactionID": 7, "TransactionDT": 12_500_000, "TransactionAmt": 12.5}
    assert client.post("/api/v1/score", json=minimal).status_code == 200
    odd = _payload(TransactionID=8, ProductCD="Z", P_emaildomain="never-seen.example")
    resp = client.post("/api/v1/score", json=odd)
    assert resp.status_code == 200, resp.text


@needs_artifacts
def test_ground_truth_is_neither_accepted_nor_returned(client):
    resp = client.post("/api/v1/score", json=_payload(isFraud=1))
    assert resp.status_code == 422
    ok = client.post("/api/v1/score", json=PAYLOAD)
    assert "isFraud" not in ok.text and "ground_truth" not in ok.text


def test_score_requires_login(api):
    resp = api.post("/api/v1/score", json=PAYLOAD)
    assert resp.status_code == 401
    bad = api.post("/api/v1/score", json=PAYLOAD, headers={"Authorization": "Bearer nope"})
    assert bad.status_code == 401


def test_invalid_payload_is_rejected(api, make_user):
    client = api
    client.headers.update(auth_header(client, make_user()))
    assert client.post("/api/v1/score", json={"TransactionID": 1}).status_code == 422
    bad_amount = _payload(TransactionAmt=-5)
    assert client.post("/api/v1/score", json=bad_amount).status_code == 422


def test_no_model_loaded_gives_503(db_session, make_user):
    store = ModelStore()
    store.error = "no active model in model_versions"
    app.dependency_overrides[get_model_store] = lambda: store
    app.dependency_overrides[get_db] = lambda: db_session
    try:
        c = TestClient(app)
        resp = c.post("/api/v1/score", json=PAYLOAD, headers=auth_header(c, make_user()))
    finally:
        app.dependency_overrides.clear()
    assert resp.status_code == 503
    assert "no active model" in resp.json()["detail"]


@pytest.mark.parametrize(
    ("p", "level"),
    [
        (0.0, "low"),
        (0.0499, "low"),
        (0.05, "medium"),
        (0.2, "high"),
        (0.5, "critical"),
        (1, "critical"),
    ],
)
def test_risk_level_bands(p, level):
    assert risk_level(p) == level
