import pytest
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.core.security import verify_password
from app.db.models import User
from app.scripts import seed_users
from tests.conftest import auth_header, login

USERS = "/api/v1/users"
NEW = {
    "email": "New.Analyst@Fraud.local",
    "full_name": "New Analyst",
    "role": "analyst",
    "password": "a-good-password",
}


def test_admin_can_list_and_create_users(api, make_user, db_session: Session):
    admin = make_user("admin")
    h = auth_header(api, admin)
    resp = api.post(USERS, json=NEW, headers=h)
    assert resp.status_code == 201, resp.text
    body = resp.json()
    assert body["email"] == "new.analyst@fraud.local" and body["role"] == "analyst"
    assert "password" not in resp.text
    listed = api.get(USERS, headers=h).json()
    assert [u["email"] for u in listed] == [admin.email, "new.analyst@fraud.local"]
    # The new user can log in with the chosen password.
    assert login(api, "new.analyst@fraud.local", "a-good-password").status_code == 200
    stored = db_session.scalar(select(User).where(User.email == "new.analyst@fraud.local"))
    assert verify_password("a-good-password", stored.password_hash)


def test_analyst_is_forbidden(api, make_user):
    h = auth_header(api, make_user("analyst"))
    assert api.get(USERS, headers=h).status_code == 403
    assert api.post(USERS, json=NEW, headers=h).status_code == 403


def test_users_api_requires_login(api):
    assert api.get(USERS).status_code == 401
    assert api.post(USERS, json=NEW).status_code == 401


def test_duplicate_email_is_a_conflict(api, make_user):
    h = auth_header(api, make_user("admin"))
    assert api.post(USERS, json=NEW, headers=h).status_code == 201
    dup = {**NEW, "email": "NEW.ANALYST@fraud.local "}
    assert api.post(USERS, json=dup, headers=h).status_code == 409


@pytest.mark.parametrize(
    "change",
    [
        {"password": "short"},
        {"password": "é" * 40},  # 80 bytes: over bcrypt's limit
        {"email": "not-an-email"},
        {"role": "superuser"},
        {"full_name": "   "},
        {"is_active": False},  # unknown field
    ],
)
def test_invalid_new_user_is_rejected(api, make_user, change):
    h = auth_header(api, make_user("admin"))
    assert api.post(USERS, json={**NEW, **change}, headers=h).status_code == 422


def test_role_change_takes_effect_immediately(api, make_user, db_session):
    user = make_user("admin")
    h = auth_header(api, user)
    assert api.get(USERS, headers=h).status_code == 200
    user.role = "analyst"
    db_session.flush()
    assert api.get(USERS, headers=h).status_code == 403  # same token, new role


# ---------------------------------------------------------------- seed script
def _run_seed(monkeypatch, db_session, **env):
    for k, v in env.items():
        monkeypatch.setenv(k, v)
    monkeypatch.setattr(seed_users, "SessionLocal", lambda: _NoClose(db_session))
    seed_users.main()


class _NoClose:
    """Use the test session inside the script's `with SessionLocal() as db` block."""

    def __init__(self, session):
        self.s = session

    def __enter__(self):
        return self.s

    def __exit__(self, *exc):
        return False


ENV = {"SEED_ADMIN_PASSWORD": "admin-password-1", "SEED_ANALYST_PASSWORD": "analyst-pass-1"}


def test_seed_creates_three_users_once(monkeypatch, db_session, capsys):
    monkeypatch.delenv("SEED_ADMIN_EMAIL", raising=False)
    _run_seed(monkeypatch, db_session, **ENV)
    _run_seed(monkeypatch, db_session, **ENV)
    users = db_session.scalars(select(User).order_by(User.id)).all()
    assert [(u.email, u.role) for u in users] == [
        ("admin@fraud.local", "admin"),
        ("analyst1@fraud.local", "analyst"),
        ("analyst2@fraud.local", "analyst"),
    ]
    assert "0 user(s) created" in capsys.readouterr().out


def test_seed_refuses_placeholder_passwords(monkeypatch, db_session):
    with pytest.raises(SystemExit, match="SEED_ADMIN_PASSWORD"):
        _run_seed(
            monkeypatch,
            db_session,
            SEED_ADMIN_PASSWORD="change-me-admin-password",
            SEED_ANALYST_PASSWORD="analyst-pass-1",
        )
    assert db_session.scalar(select(func.count()).select_from(User)) == 0
