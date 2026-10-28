"""Authentication endpoints: login, refresh, logout, current user.

The access token (15 min) is returned in the body and sent as `Authorization: Bearer`.
The refresh token (7 days) lives only in an httpOnly cookie limited to the auth endpoints,
so page scripts can never read it.
"""

from typing import Annotated

from fastapi import APIRouter, Cookie, Depends, HTTPException, Request, Response, status
from fastapi.responses import JSONResponse
from fastapi.security import OAuth2PasswordRequestForm
from sqlalchemy.orm import Session

from app.api.deps import CurrentUser
from app.core.config import get_settings
from app.core.security import create_access_token
from app.db.models import User
from app.db.session import get_db
from app.schemas.auth import TokenResponse, UserOut
from app.services import audit
from app.services.auth import (
    AuthError,
    authenticate,
    find_user_by_email,
    issue_refresh_token,
    revoke_refresh_token,
    rotate_refresh_token,
)

router = APIRouter(prefix="/auth", tags=["auth"])

REFRESH_COOKIE = "refresh_token"


def _cookie_path() -> str:
    return f"{get_settings().api_v1_prefix}/auth"


def _set_refresh_cookie(response: Response, token: str) -> None:
    s = get_settings()
    response.set_cookie(
        REFRESH_COOKIE,
        token,
        max_age=s.refresh_token_days * 24 * 3600,
        httponly=True,
        secure=s.cookie_secure,
        samesite="strict",
        path=_cookie_path(),
    )


def _clear_refresh_cookie(response: Response) -> None:
    response.delete_cookie(REFRESH_COOKIE, path=_cookie_path())


def _token_response(user: User) -> TokenResponse:
    return TokenResponse(
        access_token=create_access_token(user.id, user.role),
        expires_in=get_settings().access_token_minutes * 60,
        user=UserOut.model_validate(user),
    )


def _unauthorized(detail: str) -> HTTPException:
    return HTTPException(
        status.HTTP_401_UNAUTHORIZED, detail, headers={"WWW-Authenticate": "Bearer"}
    )


@router.post("/login", response_model=TokenResponse)
def login(
    form: Annotated[OAuth2PasswordRequestForm, Depends()],
    request: Request,
    response: Response,
    db: Annotated[Session, Depends(get_db)],
) -> TokenResponse:
    """Log in with e-mail (`username` field) and password (OAuth2 password form).

    Successful and failed logins are both written to the audit log (never the password)."""
    try:
        user = authenticate(db, form.username, form.password)
    except AuthError as exc:
        db.rollback()
        known = find_user_by_email(db, form.username)
        audit.record(
            db,
            "auth.login_failed",
            entity_type="user",
            entity_id=known.id if known else None,
            after={"email": form.username.strip().lower()[:255], "reason": exc.reason},
            request=request,
        )
        db.commit()
        raise _unauthorized(str(exc)) from exc
    token, _ = issue_refresh_token(db, user)
    audit.record(
        db, "auth.login", actor=user, entity_type="user", entity_id=user.id, request=request
    )
    db.commit()
    _set_refresh_cookie(response, token)
    return _token_response(user)


@router.post("/refresh", response_model=TokenResponse)
def refresh(
    request: Request,
    response: Response,
    db: Annotated[Session, Depends(get_db)],
    refresh_token: Annotated[str | None, Cookie()] = None,
) -> TokenResponse | JSONResponse:
    """New access token (and a rotated refresh cookie) from the refresh cookie."""
    if not refresh_token:
        raise _unauthorized("No refresh token")
    try:
        user, new_token = rotate_refresh_token(db, refresh_token)
    except AuthError as exc:
        if exc.reason == "token_reuse":
            # A copied token was replayed: every session of that user was revoked.
            audit.record(
                db,
                "auth.token_reuse",
                entity_type="user",
                entity_id=exc.user_id,
                after={"all_sessions_revoked": True},
                request=request,
            )
        db.commit()  # keeps the revocation done on token reuse
        failed = JSONResponse(
            {"detail": str(exc)},
            status_code=status.HTTP_401_UNAUTHORIZED,
            headers={"WWW-Authenticate": "Bearer"},
        )
        _clear_refresh_cookie(failed)
        return failed
    db.commit()
    _set_refresh_cookie(response, new_token)
    return _token_response(user)


@router.post("/logout", status_code=status.HTTP_204_NO_CONTENT)
def logout(
    request: Request,
    response: Response,
    db: Annotated[Session, Depends(get_db)],
    refresh_token: Annotated[str | None, Cookie()] = None,
) -> Response:
    """Revoke the refresh token of this browser and clear its cookie (works with an expired
    access token too)."""
    if refresh_token:
        user = revoke_refresh_token(db, refresh_token)
        if user is not None:
            audit.record(
                db,
                "auth.logout",
                actor=user,
                entity_type="user",
                entity_id=user.id,
                request=request,
            )
        db.commit()
    response.status_code = status.HTTP_204_NO_CONTENT
    _clear_refresh_cookie(response)
    return response


@router.get("/me", response_model=UserOut)
def me(user: CurrentUser) -> User:
    return user
