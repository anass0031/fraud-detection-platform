import json

import pytest
from fastapi.testclient import TestClient
from fraudml.registry import MODELS_DIR
from sqlalchemy import select, text
from sqlalchemy.exc import DBAPIError
from sqlalchemy.orm import Session

from app.db.models import AuditLog
from app.main import app
from app.services import audit
from app.services.model_store import activate_version, register
from tests.conftest import TEST_PASSWORD, auth_header, login
from tests.test_users import ENV, NEW, _run_seed

MODEL, ANOMALY = "lgbm-2026.10.27-1", "iforest-2026.10.27-1"


def _rows(db: Session, action: str | None = None) -> list[AuditLog]:
    q = select(AuditLog).order_by(AuditLog.id)
    if action:
        q = q.where(AuditLog.action == action)
    return list(db.scalars(q))


def _all_text(db: Session) -> str:
    return json.dumps([[r.action, r.before, r.after, r.entity_id, r.ip_address] for r in _rows(db)])


# ---------------------------------------------------------------- auth events
def test_login_success_and_failures_are_audited(api, make_user, db_session):
    user = make_user(email="a@example.com")
    make_user(email="off@example.com", active=False)
    assert login(api, "a@example.com").status_code == 200
    assert login(api, "a@example.com", "wrong-password-1").status_code == 401
    assert login(api, "ghost@example.com", "wrong-password-1").status_code == 401
    assert login(api, "off@example.com").status_code == 401

    ok = _rows(db_session, "auth.login")
    assert len(ok) == 1 and ok[0].actor_id == user.id and ok[0].entity_id == str(user.id)
    assert ok[0].ip_address == "testclient"
    failed = _rows(db_session, "auth.login_failed")
    assert [(r.after["email"], r.after["reason"]) for r in failed] == [
        ("a@example.com", "wrong_password"),
        ("ghost@example.com", "unknown_email"),
        ("off@example.com", "disabled"),
    ]
    assert all(r.actor_id is None for r in failed)
    assert failed[1].entity_id is None  # unknown e-mail: no user
    assert TEST_PASSWORD not in _all_text(db_session)
    assert "wrong-password-1" not in _all_text(db_session)


def test_logout_and_token_reuse_are_audited(api, make_user, db_session):
    user = make_user()
    stolen = login(api, user.email).cookies["refresh_token"]
    api.post("/api/v1/auth/refresh")
    assert (
        TestClient(app, cookies={"refresh_token": stolen}).post("/api/v1/auth/refresh").status_code
        == 401
    )
    reuse = _rows(db_session, "auth.token_reuse")
    assert len(reuse) == 1 and reuse[0].entity_id == str(user.id)

    login(api, user.email)
    assert api.post("/api/v1/auth/logout").status_code == 204
    out = _rows(db_session, "auth.logout")
    assert len(out) == 1 and out[0].actor_id == user.id
    assert stolen not in _all_text(db_session)


def test_refresh_is_not_audited(api, make_user, db_session):
    user = make_user()
    login(api, user.email)
    assert api.post("/api/v1/auth/refresh").status_code == 200
    assert [r.action for r in _rows(db_session)] == ["auth.login"]


# ---------------------------------------------------------------- users
def test_user_creation_is_audited_with_actor_and_snapshot(api, make_user, db_session):
    admin = make_user("admin")
    h = auth_header(api, admin)
    created = api.post("/api/v1/users", json=NEW, headers=h).json()
    rows = _rows(db_session, "user.create")
    assert len(rows) == 1
    row = rows[0]
    assert row.actor_id == admin.id and row.entity_id == str(created["id"])
    assert row.after == {
        "email": "new.analyst@fraud.local",
        "full_name": "New Analyst",
        "role": "analyst",
        "is_active": True,
    }
    assert NEW["password"] not in _all_text(db_session)
    assert "$2b$" not in _all_text(db_session)  # no bcrypt hash either

    # A refused creation writes no user.create row.
    assert api.post("/api/v1/users", json=NEW, headers=h).status_code == 409
    assert len(_rows(db_session, "user.create")) == 1


def test_seed_script_is_audited_as_system(monkeypatch, db_session):
    _run_seed(monkeypatch, db_session, **ENV)
    rows = _rows(db_session, "user.create")
    assert len(rows) == 3
    assert all(r.actor_id is None and r.after["via"] == "seed_users script" for r in rows)


# ---------------------------------------------------------------- models
@pytest.mark.skipif(not (MODELS_DIR / MODEL).exists(), reason="trained artifacts not available")
def test_model_register_and_activate_are_audited(db_session, tmp_path):
    import shutil

    for v in (MODEL, ANOMALY):
        shutil.copytree(MODELS_DIR / v, tmp_path / v)
    shutil.copytree(MODELS_DIR / MODEL, tmp_path / "lgbm-test-2")
    register(db_session, MODEL, ANOMALY, activate=True, root=tmp_path)
    register(db_session, "lgbm-test-2", ANOMALY, root=tmp_path)
    activate_version(db_session, "lgbm-test-2")

    assert [r.action for r in _rows(db_session)] == [
        "model.register",
        "model.activate",
        "model.register",
        "model.activate",
    ]
    last = _rows(db_session, "model.activate")[-1]
    assert last.actor_id is None
    assert last.before == {"status": "candidate", "previous_active": MODEL}
    assert last.after["previous_archived"] == MODEL
    # Activating the already active version changes nothing and writes nothing.
    activate_version(db_session, "lgbm-test-2")
    assert len(_rows(db_session, "model.activate")) == 2


# ---------------------------------------------------------------- append-only + safety
@pytest.mark.parametrize(
    "sql",
    [
        "UPDATE audit_logs SET action = 'tampered'",
        "DELETE FROM audit_logs",
        "TRUNCATE audit_logs",
    ],
)
def test_audit_rows_cannot_be_changed_or_removed(db_session, sql):
    audit.record(db_session, "test.event", entity_type="test", entity_id=1)
    with pytest.raises(DBAPIError, match="append-only"):
        with db_session.begin_nested():
            db_session.execute(text(sql))
    assert [r.action for r in _rows(db_session)] == ["test.event"]


def test_audit_row_is_part_of_the_callers_transaction(db_session):
    with db_session.begin_nested() as sp:
        audit.record(db_session, "test.event")
        sp.rollback()
    assert _rows(db_session) == []


@pytest.mark.parametrize(
    "data", [{"password": "x"}, {"new_token": "x"}, {"nested": {"password_hash": "x"}}]
)
def test_sensitive_fields_are_refused(db_session, data):
    with pytest.raises(audit.SensitiveAuditData):
        audit.record(db_session, "test.event", after=data)


def test_missing_before_after_are_sql_null(db_session):
    audit.record(db_session, "test.event")
    n = db_session.execute(
        text("SELECT count(*) FROM audit_logs WHERE before IS NULL AND after IS NULL")
    ).scalar()
    assert n == 1
