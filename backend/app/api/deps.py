"""Shared FastAPI dependencies: the current user and role checks."""

from __future__ import annotations

from collections.abc import Callable
from typing import Annotated

from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer
from sqlalchemy.orm import Session

from app.core.config import get_settings
from app.core.security import InvalidToken, decode_access_token
from app.db.models import User
from app.db.session import get_db

# auto_error=False: we return our own 401 (same message for every failure).
oauth2_scheme = OAuth2PasswordBearer(
    tokenUrl=f"{get_settings().api_v1_prefix}/auth/login", auto_error=False
)


def _unauthorized(detail: str = "Not authenticated") -> HTTPException:
    return HTTPException(
        status.HTTP_401_UNAUTHORIZED, detail, headers={"WWW-Authenticate": "Bearer"}
    )


def get_current_user(
    token: Annotated[str | None, Depends(oauth2_scheme)],
    db: Annotated[Session, Depends(get_db)],
) -> User:
    """The user of a valid access token. Re-read from the DB on every request, so a
    deactivated user or a changed role takes effect immediately."""
    if not token:
        raise _unauthorized()
    try:
        claims = decode_access_token(token)
        user_id = int(claims["sub"])
    except (InvalidToken, ValueError) as exc:
        raise _unauthorized("Invalid or expired token") from exc
    user = db.get(User, user_id)
    if user is None or not user.is_active:
        raise _unauthorized("Invalid or expired token")
    return user


CurrentUser = Annotated[User, Depends(get_current_user)]


def require_role(*roles: str) -> Callable[[User], User]:
    """Dependency factory: the current user must have one of `roles` (403 otherwise)."""

    def check(user: CurrentUser) -> User:
        if user.role not in roles:
            raise HTTPException(status.HTTP_403_FORBIDDEN, "Not allowed for your role")
        return user

    return check
