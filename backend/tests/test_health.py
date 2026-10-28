import datetime as dt
from types import SimpleNamespace

from fastapi.testclient import TestClient
from sqlalchemy.exc import OperationalError

from app.db.session import get_db
from app.main import app
from app.services.model_store import ModelStore, get_model_store

client = TestClient(app)


def _store(loaded: bool) -> ModelStore:
    store = ModelStore()
    if loaded:
        store.bundle = SimpleNamespace(
            version="lgbm-x",
            anomaly_version="iforest-x",
            loaded_at=dt.datetime(2026, 10, 28, tzinfo=dt.UTC),
        )
        store.error = None
    else:
        store.error = "no active model in model_versions"
    return store


def _get(store: ModelStore, db=None):
    app.dependency_overrides[get_model_store] = lambda: store
    if db is not None:
        app.dependency_overrides[get_db] = lambda: db
    try:
        return client.get("/api/v1/health")
    finally:
        app.dependency_overrides.clear()


def test_health_ok_when_database_and_model_are_up():
    resp = _get(_store(loaded=True))
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "ok"
    assert body["database"] == "ok"
    assert body["version"] == "0.1.0"
    assert body["model"]["status"] == "loaded"
    assert body["model"]["version"] == "lgbm-x"
    assert body["model"]["anomaly_version"] == "iforest-x"


def test_health_degraded_without_model():
    resp = _get(_store(loaded=False))
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "degraded"
    assert body["model"] == {
        "status": "not_loaded",
        "version": None,
        "anomaly_version": None,
        "loaded_at": None,
        "error": "no active model in model_versions",
    }


class _BrokenSession:
    def execute(self, *_args, **_kwargs):
        raise OperationalError("SELECT 1", {}, Exception("db down"))


def test_health_returns_503_when_database_is_down():
    resp = _get(_store(loaded=True), db=_BrokenSession())
    assert resp.status_code == 503
    assert resp.json()["database"] == "error"
    assert resp.json()["status"] == "degraded"
