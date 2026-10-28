"""Audit trail: one append-only row per state-changing action.

Call `record(...)` in the same database transaction as the change itself, before the commit,
so a change and its audit row are always saved (or rolled back) together.

Action names: `<entity>.<verb>`, e.g. auth.login, auth.login_failed, auth.logout,
user.create, model.register, model.activate, case.create (system), case.claim, case.assign,
case.decide, settings.update, replay.day (system: one replayed test day).
"""

from __future__ import annotations

from typing import Any

from fastapi import Request
from sqlalchemy.orm import Session

from app.db.models import AuditLog, User

# Never stored, whatever the caller passes (defence in depth).
SENSITIVE_KEYS = ("password", "token", "secret", "hash")


class SensitiveAuditData(ValueError):
    pass


def _check(data: dict[str, Any] | None) -> None:
    if not data:
        return
    for key, value in data.items():
        if any(word in key.lower() for word in SENSITIVE_KEYS):
            raise SensitiveAuditData(f"audit data must not contain {key!r}")
        if isinstance(value, dict):
            _check(value)


def client_ip(request: Request | None) -> str | None:
    return request.client.host if request is not None and request.client else None


def record(
    db: Session,
    action: str,
    *,
    actor: User | None = None,
    entity_type: str | None = None,
    entity_id: Any = None,
    before: dict[str, Any] | None = None,
    after: dict[str, Any] | None = None,
    request: Request | None = None,
) -> AuditLog:
    """Add an audit row to the session (written with the caller's commit).

    actor=None means a system action (e.g. a script).
    """
    _check(before)
    _check(after)
    row = AuditLog(
        actor_id=actor.id if actor is not None else None,
        action=action,
        entity_type=entity_type,
        entity_id=None if entity_id is None else str(entity_id),
        before=before,
        after=after,
        ip_address=client_ip(request),
    )
    db.add(row)
    db.flush()
    return row


def user_snapshot(user: User) -> dict[str, Any]:
    """The audited view of a user (no password hash)."""
    return {
        "email": user.email,
        "full_name": user.full_name,
        "role": user.role,
        "is_active": user.is_active,
    }
