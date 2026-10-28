"""Login checks and refresh-token lifecycle (issue, rotate with reuse detection, revoke)."""

from __future__ import annotations

import datetime as dt

from sqlalchemy import select, update
from sqlalchemy.orm import Session

from app.core.config import get_settings
from app.core.security import hash_token, new_refresh_token, verify_dummy, verify_password
from app.db.models import RefreshToken, User


class AuthError(Exception):
    """Login or refresh refused. The message is safe to show; `reason` is for the audit log."""

    def __init__(self, message: str, reason: str = "refused", user_id: int | None = None):
        super().__init__(message)
        self.reason = reason
        self.user_id = user_id


def _now() -> dt.datetime:
    return dt.datetime.now(dt.UTC)


def find_user_by_email(db: Session, email: str) -> User | None:
    return db.scalar(select(User).where(User.email == email.strip().lower()))


def authenticate(db: Session, email: str, password: str) -> User:
    """The user for these credentials. Same error for unknown e-mail and wrong password."""
    user = find_user_by_email(db, email)
    if user is None:
        verify_dummy(password)  # similar timing whether or not the e-mail exists
        raise AuthError("Incorrect e-mail or password", "unknown_email")
    if not verify_password(password, user.password_hash):
        raise AuthError("Incorrect e-mail or password", "wrong_password")
    if not user.is_active:
        raise AuthError("This account is disabled", "disabled")
    user.last_login_at = _now()
    return user


def issue_refresh_token(db: Session, user: User) -> tuple[str, RefreshToken]:
    token = new_refresh_token()
    row = RefreshToken(
        user_id=user.id,
        token_hash=hash_token(token),
        expires_at=_now() + dt.timedelta(days=get_settings().refresh_token_days),
    )
    db.add(row)
    db.flush()
    return token, row


def revoke_all(db: Session, user_id: int) -> None:
    db.execute(
        update(RefreshToken)
        .where(RefreshToken.user_id == user_id, RefreshToken.revoked_at.is_(None))
        .values(revoked_at=_now())
    )


def rotate_refresh_token(db: Session, token: str) -> tuple[User, str]:
    """Exchange a valid refresh token for a new one (the old one is revoked).

    Re-using a revoked token means it was copied: every token of that user is revoked.
    The caller must commit even when AuthError is raised for reuse, so the revocation sticks.
    """
    row = db.scalar(select(RefreshToken).where(RefreshToken.token_hash == hash_token(token)))
    if row is None:
        raise AuthError("Invalid refresh token")
    if row.revoked_at is not None:
        revoke_all(db, row.user_id)
        raise AuthError(
            "Refresh token already used; please log in again", "token_reuse", row.user_id
        )
    if row.expires_at <= _now():
        raise AuthError("Refresh token expired; please log in again")
    user = row.user
    if not user.is_active:
        raise AuthError("This account is disabled")
    new_token, new_row = issue_refresh_token(db, user)
    row.revoked_at = _now()
    row.replaced_by_id = new_row.id
    return user, new_token


def revoke_refresh_token(db: Session, token: str) -> User | None:
    """Logout: revoke this refresh token if it is still active. Returns its user."""
    row = db.scalar(select(RefreshToken).where(RefreshToken.token_hash == hash_token(token)))
    if row is None:
        return None
    if row.revoked_at is None:
        row.revoked_at = _now()
    return row.user
