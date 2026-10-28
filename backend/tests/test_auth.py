import datetime as dt

import jwt
import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.core.config import Settings, get_settings
from app.core.security import (
    create_access_token,
    hash_password,
    password_problem,
    verify_password,
)
from app.db.models import RefreshToken
from app.main import app
from tests.conftest import TEST_PASSWORD, login

ME = "/api/v1/auth/me"
REFRESH = "/api/v1/auth/refresh"
LOGOUT = "/api/v1/auth/logout"


def _bearer(token: str) -> dict[str, str]:
    return {"Authorization": f"Bearer {token}"}


# ---------------------------------------------------------------- passwords and settings
def test_password_hash_round_trip():
    h = hash_password("a-long-password")
    assert h.startswith("$2") and "a-long-password" not in h
    assert verify_password("a-long-password", h)
    assert not verify_password("another-password", h)


def test_password_rules():
    assert password_problem("short") is not None
    assert password_problem("x" * 73) is not None  # over bcrypt's 72-byte limit
    assert password_problem("é" * 40) is not None  # 80 bytes in UTF-8
    assert password_problem("long-enough-password") is None


@pytest.mark.parametrize("secret", ["too-short", "change-me-generate-a-long-random-secret"])
def test_weak_jwt_secret_is_refused(secret):
    with pytest.raises(ValidationError):
        Settings(jwt_secret=secret)


# ---------------------------------------------------------------- login
def test_login_returns_access_token_and_httponly_refresh_cookie(api, make_user):
    user = make_user("admin", email="boss@example.com")
    resp = login(api, "Boss@Example.com ")  # e-mail is case-insensitive, spaces trimmed
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["token_type"] == "bearer"
    assert body["expires_in"] == 15 * 60
    assert body["user"]["email"] == "boss@example.com" and body["user"]["role"] == "admin"
    assert "password_hash" not in body["user"]
    cookie = resp.headers["set-cookie"]
    assert "refresh_token=" in cookie and "HttpOnly" in cookie
    assert "Path=/api/v1/auth" in cookie and "SameSite=strict" in cookie
    claims = jwt.decode(body["access_token"], get_settings().jwt_secret, algorithms=["HS256"])
    assert claims["sub"] == str(user.id) and claims["role"] == "admin"
    assert user.last_login_at is not None


def test_wrong_password_and_unknown_email_look_the_same(api, make_user):
    make_user(email="known@example.com")
    wrong = login(api, "known@example.com", "wrong-password-123")
    unknown = login(api, "nobody@example.com", "wrong-password-123")
    assert wrong.status_code == unknown.status_code == 401
    assert wrong.json() == unknown.json() == {"detail": "Incorrect e-mail or password"}


def test_disabled_account_cannot_log_in(api, make_user):
    make_user(email="gone@example.com", active=False)
    resp = login(api, "gone@example.com")
    assert resp.status_code == 401
    assert resp.json()["detail"] == "This account is disabled"


# ---------------------------------------------------------------- access tokens
def test_me_requires_a_valid_token(api, make_user):
    user = make_user()
    token = login(api, user.email).json()["access_token"]
    assert api.get(ME, headers=_bearer(token)).json()["id"] == user.id
    assert api.get(ME).status_code == 401
    assert api.get(ME, headers=_bearer(token[:-2] + "xx")).status_code == 401  # tampered
    forged = jwt.encode({"sub": str(user.id), "type": "access"}, "x" * 40, algorithm="HS256")
    assert api.get(ME, headers=_bearer(forged)).status_code == 401  # other secret


def test_expired_access_token_is_refused(api, make_user):
    user = make_user()
    old = dt.datetime.now(dt.UTC) - dt.timedelta(minutes=16)
    token = create_access_token(user.id, user.role, now=old)
    resp = api.get(ME, headers=_bearer(token))
    assert resp.status_code == 401
    assert resp.headers["www-authenticate"] == "Bearer"


def test_deactivated_user_is_refused_with_a_still_valid_token(api, make_user, db_session):
    user = make_user()
    token = login(api, user.email).json()["access_token"]
    user.is_active = False
    db_session.flush()
    assert api.get(ME, headers=_bearer(token)).status_code == 401


# ---------------------------------------------------------------- refresh tokens
def test_refresh_rotates_the_token(api, make_user, db_session: Session):
    user = make_user()
    first = login(api, user.email).cookies["refresh_token"]
    resp = api.post(REFRESH)
    assert resp.status_code == 200, resp.text
    second = resp.cookies["refresh_token"]
    assert second != first
    assert api.get(ME, headers=_bearer(resp.json()["access_token"])).status_code == 200
    rows = db_session.scalars(select(RefreshToken).order_by(RefreshToken.id)).all()
    assert len(rows) == 2
    assert rows[0].revoked_at is not None and rows[0].replaced_by_id == rows[1].id
    assert rows[1].revoked_at is None
    assert first not in {r.token_hash for r in rows}  # only hashes are stored


def test_reusing_an_old_refresh_token_revokes_the_whole_family(api, make_user, db_session):
    user = make_user()
    stolen = login(api, user.email).cookies["refresh_token"]
    assert api.post(REFRESH).status_code == 200  # the real user rotates

    attacker = TestClient(app, cookies={"refresh_token": stolen})
    resp = attacker.post(REFRESH)
    assert resp.status_code == 401
    assert "already used" in resp.json()["detail"]
    # The real user's current token is revoked too: they must log in again.
    assert api.post(REFRESH).status_code == 401
    active = db_session.scalars(select(RefreshToken).where(RefreshToken.revoked_at.is_(None))).all()
    assert active == []


def test_expired_refresh_token_is_refused(api, make_user, db_session):
    user = make_user()
    login(api, user.email)
    row = db_session.scalar(select(RefreshToken))
    row.expires_at = dt.datetime.now(dt.UTC) - dt.timedelta(seconds=1)
    db_session.flush()
    resp = api.post(REFRESH)
    assert resp.status_code == 401 and "expired" in resp.json()["detail"]


def test_refresh_without_cookie_is_refused(api):
    assert api.post(REFRESH).status_code == 401


def test_refresh_refused_for_disabled_user(api, make_user, db_session):
    user = make_user()
    login(api, user.email)
    user.is_active = False
    db_session.flush()
    assert api.post(REFRESH).status_code == 401


# ---------------------------------------------------------------- logout
def test_logout_revokes_the_refresh_token_and_clears_the_cookie(api, make_user):
    user = make_user()
    token = login(api, user.email).cookies["refresh_token"]
    resp = api.post(LOGOUT)
    assert resp.status_code == 204
    assert (
        'refresh_token=""' in resp.headers["set-cookie"]
        or "Max-Age=0" in resp.headers["set-cookie"]
    )
    again = TestClient(app, cookies={"refresh_token": token})
    assert again.post(REFRESH).status_code == 401


def test_logout_without_cookie_is_harmless(api):
    assert api.post(LOGOUT).status_code == 204


def test_password_is_never_returned(api, make_user):
    user = make_user()
    resp = login(api, user.email)
    assert TEST_PASSWORD not in resp.text
    assert "password" not in api.get(ME, headers=_bearer(resp.json()["access_token"])).text
