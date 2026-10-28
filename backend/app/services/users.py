"""Creating and listing users (shared by the users API and the seed script)."""

from __future__ import annotations

import re

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.core.security import hash_password, password_problem
from app.db.models import USER_ROLES, User

# Simple shape check (no deliverability check, so demo domains like fraud.local are allowed).
EMAIL_RE = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")


class UserError(ValueError):
    """The user cannot be created (the message is safe to show)."""


class DuplicateEmail(UserError):
    pass


def normalise_email(email: str) -> str:
    return email.strip().lower()


def create_user(db: Session, email: str, full_name: str, role: str, password: str) -> User:
    email = normalise_email(email)
    if not EMAIL_RE.match(email):
        raise UserError("invalid e-mail address")
    if role not in USER_ROLES:
        raise UserError(f"role must be one of {USER_ROLES}")
    if not full_name.strip():
        raise UserError("full name is required")
    problem = password_problem(password)
    if problem:
        raise UserError(problem)
    if db.scalar(select(User.id).where(User.email == email)) is not None:
        raise DuplicateEmail(f"a user with e-mail {email} already exists")
    user = User(
        email=email,
        full_name=full_name.strip(),
        password_hash=hash_password(password),
        role=role,
        is_active=True,
    )
    db.add(user)
    db.flush()
    return user


def list_users(db: Session) -> list[User]:
    return list(db.scalars(select(User).order_by(User.id)))
