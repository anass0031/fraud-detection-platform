"""Password hashing (bcrypt), access tokens (JWT) and refresh tokens (random, stored hashed)."""

from __future__ import annotations

import datetime as dt
import hashlib
import secrets
import uuid
from typing import Any

import bcrypt
import jwt

from app.core.config import get_settings

PASSWORD_MIN_LENGTH = 10
PASSWORD_MAX_BYTES = 72  # bcrypt ignores anything longer: refuse instead of truncating
ACCESS_TOKEN_TYPE = "access"


class InvalidToken(Exception):
    """Missing, expired, tampered or wrong-type token."""


# ---------------------------------------------------------------- passwords
def password_problem(password: str) -> str | None:
    """Why a new password is not acceptable, or None when it is."""
    if len(password) < PASSWORD_MIN_LENGTH:
        return f"password must have at least {PASSWORD_MIN_LENGTH} characters"
    if len(password.encode("utf-8")) > PASSWORD_MAX_BYTES:
        return f"password must be at most {PASSWORD_MAX_BYTES} bytes"
    return None


def hash_password(password: str) -> str:
    rounds = get_settings().bcrypt_rounds
    return bcrypt.hashpw(password.encode("utf-8"), bcrypt.gensalt(rounds=rounds)).decode()


def verify_password(password: str, password_hash: str) -> bool:
    data = password.encode("utf-8")
    if len(data) > PASSWORD_MAX_BYTES:
        return False
    return bcrypt.checkpw(data, password_hash.encode())


_dummy_hash: str | None = None


def verify_dummy(password: str) -> None:
    """Spend the same time as a real check when the e-mail is unknown (no user enumeration)."""
    global _dummy_hash
    if _dummy_hash is None:
        _dummy_hash = hash_password(secrets.token_urlsafe(16))
    verify_password(password, _dummy_hash)


# ---------------------------------------------------------------- access tokens (JWT)
def create_access_token(user_id: int, role: str, now: dt.datetime | None = None) -> str:
    s = get_settings()
    now = now or dt.datetime.now(dt.UTC)
    claims: dict[str, Any] = {
        "sub": str(user_id),
        "role": role,
        "type": ACCESS_TOKEN_TYPE,
        "iat": now,
        "exp": now + dt.timedelta(minutes=s.access_token_minutes),
        "jti": uuid.uuid4().hex,
    }
    return jwt.encode(claims, s.jwt_secret, algorithm=s.jwt_algorithm)


def decode_access_token(token: str) -> dict[str, Any]:
    s = get_settings()
    try:
        claims = jwt.decode(
            token,
            s.jwt_secret,
            algorithms=[s.jwt_algorithm],
            options={"require": ["sub", "exp", "iat", "type"]},
        )
    except jwt.PyJWTError as exc:
        raise InvalidToken(str(exc)) from exc
    if claims.get("type") != ACCESS_TOKEN_TYPE:
        raise InvalidToken("not an access token")
    return claims


# ---------------------------------------------------------------- refresh tokens
def new_refresh_token() -> str:
    """An opaque random token (only its hash is stored)."""
    return secrets.token_urlsafe(48)


def hash_token(token: str) -> str:
    return hashlib.sha256(token.encode()).hexdigest()
