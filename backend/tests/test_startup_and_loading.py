"""Paths the API relies on but no endpoint test reaches: startup model loading, the drift
reference loader and the Parquet period reader."""

import datetime as dt
import json

import pandas as pd
from fastapi.testclient import TestClient
from fraudml import config
from fraudml.features.pipeline import RAW_INPUT_COLUMNS
from fraudml.monitoring.drift import SCORE, build_reference
from sqlalchemy.exc import OperationalError
from sqlalchemy.orm import Session

from app import main
from app.services import monitoring
from app.services.model_store import ModelStore
from app.services.monitoring import compute_day
from app.services.transactions_loader import read_period


class _Session:
    """Context manager standing in for SessionLocal() with a given session."""

    def __init__(self, session):
        self.session = session

    def __enter__(self):
        return self.session

    def __exit__(self, *exc):
        return False


def test_startup_without_an_active_model_still_serves(db_session: Session, monkeypatch):
    store = ModelStore()
    monkeypatch.setattr(main, "store", store)
    monkeypatch.setattr(main, "SessionLocal", lambda: _Session(db_session))
    with TestClient(main.app) as client:  # runs the lifespan (startup)
        assert client.get("/api/v1/health").status_code in (200, 503)
    assert store.bundle is None
    assert store.error == "no active model in model_versions"


def test_startup_survives_an_unreachable_database(monkeypatch):
    store = ModelStore()

    def broken():
        raise OperationalError("SELECT 1", {}, Exception("connection refused"))

    monkeypatch.setattr(main, "store", store)
    monkeypatch.setattr(main, "SessionLocal", broken)
    with TestClient(main.app):
        pass
    assert store.bundle is None
    assert store.error == "database unavailable at startup: OperationalError"


def test_drift_reference_loader(tmp_path, monkeypatch):
    monkeypatch.setattr(monitoring, "MONITORING_DIR", tmp_path)
    monitoring.load_drift_reference.cache_clear()
    assert monitoring.load_drift_reference("nope") is None

    data = build_reference(pd.DataFrame({"amt_log": [1.0, 2.0, 3.0]}), set(), [0.1, 0.2, 0.3])
    (tmp_path / "drift_reference-m1.json").write_text(json.dumps(data), encoding="utf-8")
    monitoring.load_drift_reference.cache_clear()
    refs = monitoring.load_drift_reference("m1")
    assert set(refs) == {"amt_log", SCORE}
    monitoring.load_drift_reference.cache_clear()


def test_compute_day_without_a_reference_stores_metrics_only(
    db_session: Session, make_prediction, monkeypatch
):
    p = make_prediction(when=dt.datetime(2018, 4, 28, 9), ground_truth=True)
    p.transaction.ground_truth_revealed_at = dt.datetime.now(dt.UTC)
    db_session.flush()
    monkeypatch.setattr(monitoring, "load_drift_reference", lambda version: None)

    class Model:
        model_version_id = p.model_version_id
        version = "no-reference"

    assert compute_day(db_session, Model, dt.date(2018, 4, 28)) == (True, 0)


def test_read_period_filters_and_orders(tmp_path):
    rows = pd.DataFrame({c: [None] * 4 for c in RAW_INPUT_COLUMNS})
    rows[config.ID_COL] = [4, 3, 2, 1]
    rows[config.TIME_COL] = [300, 200, 200, 100]
    rows[config.DATETIME_COL] = pd.to_datetime(
        ["2018-04-29 00:00", "2018-04-28 12:00", "2018-04-28 12:00", "2018-04-27 23:59"]
    )
    rows[config.TARGET] = [0, 1, 0, 0]
    path = tmp_path / "train.parquet"
    rows.to_parquet(path)

    df = read_period("2018-04-28", "2018-04-29", path=path)
    # [start, end): the 27th and the 29th are excluded; ties on time ordered by id.
    assert df[config.ID_COL].tolist() == [2, 3]
    assert read_period(None, None, path=path)[config.ID_COL].tolist() == [1, 2, 3, 4]


def test_read_period_empty_range(tmp_path):
    rows = pd.DataFrame({c: [None] for c in RAW_INPUT_COLUMNS})
    rows[config.ID_COL] = [1]
    rows[config.TIME_COL] = [100]
    rows[config.DATETIME_COL] = pd.to_datetime(["2018-04-27"])
    rows[config.TARGET] = [0]
    path = tmp_path / "train.parquet"
    rows.to_parquet(path)
    assert read_period("2018-05-01", None, path=path).empty
