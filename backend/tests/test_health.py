from fastapi.testclient import TestClient
from sqlalchemy.exc import OperationalError

from app.db.session import get_db
from app.main import app

client = TestClient(app)


def test_health_ok_when_database_is_up():
    resp = client.get("/api/v1/health")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok", "database": "ok", "version": "0.1.0"}


class _BrokenSession:
    def execute(self, *_args, **_kwargs):
        raise OperationalError("SELECT 1", {}, Exception("db down"))


def test_health_returns_503_when_database_is_down():
    app.dependency_overrides[get_db] = lambda: _BrokenSession()
    try:
        resp = client.get("/api/v1/health")
    finally:
        app.dependency_overrides.clear()
    assert resp.status_code == 503
    assert resp.json()["database"] == "error"
    assert resp.json()["status"] == "degraded"
