"""Minimal user management (admin only): list and create. Edit/deactivate are not in the MVP."""

from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Request, status
from sqlalchemy.orm import Session

from app.api.deps import require_role
from app.db.models import User
from app.db.session import get_db
from app.schemas.auth import UserOut
from app.schemas.users import UserCreate
from app.services import audit
from app.services.users import DuplicateEmail, UserError, create_user, list_users

router = APIRouter(prefix="/users", tags=["users"])

AdminUser = Annotated[User, Depends(require_role("admin"))]


@router.get("", response_model=list[UserOut])
def get_users(_admin: AdminUser, db: Annotated[Session, Depends(get_db)]) -> list[User]:
    return list_users(db)


@router.post("", response_model=UserOut, status_code=status.HTTP_201_CREATED)
def post_user(
    body: UserCreate,
    admin: AdminUser,
    request: Request,
    db: Annotated[Session, Depends(get_db)],
) -> User:
    try:
        user = create_user(db, body.email, body.full_name, body.role, body.password)
    except DuplicateEmail as exc:
        db.rollback()
        raise HTTPException(status.HTTP_409_CONFLICT, str(exc)) from exc
    except UserError as exc:
        db.rollback()
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_CONTENT, str(exc)) from exc
    audit.record(
        db,
        "user.create",
        actor=admin,
        entity_type="user",
        entity_id=user.id,
        after=audit.user_snapshot(user),
        request=request,
    )
    db.commit()
    return user
