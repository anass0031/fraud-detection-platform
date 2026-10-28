import shutil
from collections.abc import Generator
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from fraudml.registry import MODELS_DIR
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.db.models import AuditLog, ModelVersion
from app.db.session import get_db
from app.main import app
from app.services import settings as settings_svc
from app.services.model_store import ModelStore, get_model_store, register
from tests.conftest import auth_header
from tests.test_score import PAYLOAD

MODEL, ANOMALY = "lgbm-2026.10.27-1", "iforest-2026.10.27-1"
URL = "/api/v1/models"

pytestmark = pytest.mark.skipif(
    not (MODELS_DIR / MODEL).exists() or not (MODELS_DIR / ANOMALY).exists(),
    reason="trained artifacts not available",
)


@pytest.fixture
def models_root(tmp_path: Path) -> Path:
    for v in (MODEL, ANOMALY):
        shutil.copytree(MODELS_DIR / v, tmp_path / v)
    shutil.copytree(MODELS_DIR / MODEL, tmp_path / "lgbm-test-2")
    shutil.copytree(MODELS_DIR / MODEL, tmp_path / "lgbm-test-broken")
    return tmp_path


@pytest.fixture
def store(db_session: Session, models_root: Path) -> ModelStore:
    """Champion active and loaded; two candidates registered (one will be broken)."""
    register(db_session, MODEL, ANOMALY, activate=True, root=models_root)
    register(db_session, "lgbm-test-2", ANOMALY, root=models_root)
    register(db_session, "lgbm-test-broken", ANOMALY, root=models_root)
    (models_root / "lgbm-test-broken" / "model.txt").unlink()  # breaks after registration
    s = ModelStore(root=models_root)
    assert s.load_active(db_session).version == MODEL
    return s


@pytest.fixture
def client(db_session: Session, store: ModelStore) -> Generator[TestClient, None, None]:
    app.dependency_overrides[get_db] = lambda: db_session
    app.dependency_overrides[get_model_store] = lambda: store
    try:
        yield TestClient(app)
    finally:
        app.dependency_overrides.clear()


@pytest.fixture
def admin(make_user):
    return make_user("admin")


def row(db: Session, version: str) -> ModelVersion:
    return db.scalar(select(ModelVersion).where(ModelVersion.version == version))


def audits(db: Session, action: str) -> list[AuditLog]:
    return list(db.scalars(select(AuditLog).where(AuditLog.action == action).order_by(AuditLog.id)))


def test_admin_only(client, make_user):
    assert client.get(URL).status_code == 401
    analyst = auth_header(client, make_user("analyst"))
    assert client.get(URL, headers=analyst).status_code == 403
    assert client.post(f"{URL}/1/activate", headers=analyst).status_code == 403


def test_list_puts_the_active_model_first(client, admin):
    body = client.get(URL, headers=auth_header(client, admin)).json()
    assert [(m["version"], m["status"], m["serving"]) for m in body] == [
        (MODEL, "active", True),
        ("lgbm-test-broken", "candidate", False),
        ("lgbm-test-2", "candidate", False),
    ]
    assert body[0]["metrics"]["validation"]["overall"]["pr_auc"] > 0.3
    assert "comparison" in body[0]["metrics"]


def test_activate_swaps_model_resets_override_and_audits(client, db_session, store, admin):
    settings_svc.update_setting(
        db_session,
        settings_svc.THRESHOLDS,
        {"alert_threshold": 30.0, "risk_bands": [0.05, 0.2, 0.5]},
        admin,
    )
    db_session.flush()
    h = auth_header(client, admin)
    target = row(db_session, "lgbm-test-2")

    resp = client.post(f"{URL}/{target.id}/activate", headers=h)
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["model"]["status"] == "active" and body["model"]["serving"] is True
    assert body["threshold_override_reset"] is True

    assert row(db_session, MODEL).status == "archived"
    assert store.bundle.version == "lgbm-test-2"  # hot reload, no restart
    value, _ = settings_svc.get_setting(db_session, settings_svc.THRESHOLDS)
    assert value == {"alert_threshold": None, "risk_bands": [0.05, 0.2, 0.5]}

    (activate,) = [a for a in audits(db_session, "model.activate") if a.actor_id == admin.id]
    assert activate.after["previous_archived"] == MODEL
    assert activate.ip_address is not None
    reset = audits(db_session, "settings.update")[-1]
    assert reset.before["alert_threshold"] == 30.0 and reset.after["alert_threshold"] is None

    # The scoring API uses the new model right away.
    scored = client.post("/api/v1/score", json=PAYLOAD, headers=h).json()
    assert scored["model_version"] == "lgbm-test-2"


def test_activate_without_override_resets_nothing(client, db_session, admin):
    target = row(db_session, "lgbm-test-2")
    body = client.post(f"{URL}/{target.id}/activate", headers=auth_header(client, admin)).json()
    assert body["threshold_override_reset"] is False
    assert audits(db_session, "settings.update") == []


def test_broken_artifact_changes_nothing(client, db_session, store, admin):
    broken = row(db_session, "lgbm-test-broken")
    resp = client.post(f"{URL}/{broken.id}/activate", headers=auth_header(client, admin))
    assert resp.status_code == 409
    assert "cannot be loaded" in resp.json()["detail"]
    assert row(db_session, MODEL).status == "active"
    assert row(db_session, "lgbm-test-broken").status == "candidate"
    assert store.bundle.version == MODEL  # still serving the champion
    assert [a.actor_id for a in audits(db_session, "model.activate")] == [None]  # only the seed


def test_rollback_to_an_archived_model(client, db_session, store, admin):
    h = auth_header(client, admin)
    client.post(f"{URL}/{row(db_session, 'lgbm-test-2').id}/activate", headers=h)
    resp = client.post(f"{URL}/{row(db_session, MODEL).id}/activate", headers=h)
    assert resp.status_code == 200
    assert store.bundle.version == MODEL
    assert row(db_session, "lgbm-test-2").status == "archived"


def test_archive_rules(client, db_session, admin):
    h = auth_header(client, admin)
    candidate = row(db_session, "lgbm-test-2")
    resp = client.post(f"{URL}/{candidate.id}/archive", headers=h)
    assert resp.status_code == 200 and resp.json()["status"] == "archived"
    (a,) = audits(db_session, "model.archive")
    assert (a.actor_id, a.after["version"]) == (admin.id, "lgbm-test-2")

    active = row(db_session, MODEL)
    assert client.post(f"{URL}/{active.id}/archive", headers=h).status_code == 409
    assert client.post(f"{URL}/999999/archive", headers=h).status_code == 404
    assert client.get(f"{URL}/999999", headers=h).status_code == 404


def test_startup_survives_a_broken_active_model(db_session: Session, store, models_root):
    from app.services.model_store import activate_version

    activate_version(db_session, "lgbm-test-broken")  # bypasses the API's load check
    fresh = ModelStore(root=models_root)
    assert fresh.load_active(db_session) is None  # no crash: the API starts without a model
    assert "LightGBMError" in fresh.error
