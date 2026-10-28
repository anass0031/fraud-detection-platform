"""Model versions (admin only): list, read, activate (champion/challenger promotion), archive.

Candidates come from the offline retrain script + `register_model`; there is no API to train.
"""

from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Request, status
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.api.deps import require_role
from app.db.models import ModelVersion, User
from app.db.session import get_db
from app.schemas.models import ActivateResponse, ModelOut
from app.services.model_admin import ModelLoadError, promote
from app.services.model_store import ModelStore, RegistryError, archive_version, get_model_store

router = APIRouter(prefix="/models", tags=["models"])

AdminUser = Annotated[User, Depends(require_role("admin"))]
Db = Annotated[Session, Depends(get_db)]
Store = Annotated[ModelStore, Depends(get_model_store)]

STATUS_ORDER = {"active": 0, "candidate": 1, "archived": 2}


def out(row: ModelVersion, store: ModelStore) -> ModelOut:
    m = ModelOut.model_validate(row)
    m.serving = store.bundle is not None and store.bundle.version == row.version
    return m


def get_row(db: Session, model_id: int) -> ModelVersion:
    row = db.get(ModelVersion, model_id)
    if row is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Model version not found")
    return row


@router.get("", response_model=list[ModelOut])
def list_models(_admin: AdminUser, db: Db, store: Store) -> list[ModelOut]:
    """All versions: active first, then candidates, then archived (newest first)."""
    rows = db.scalars(select(ModelVersion)).all()
    rows = sorted(rows, key=lambda r: (STATUS_ORDER.get(r.status, 9), -r.id))
    return [out(r, store) for r in rows]


@router.get("/{model_id}", response_model=ModelOut)
def get_model(model_id: int, _admin: AdminUser, db: Db, store: Store) -> ModelOut:
    return out(get_row(db, model_id), store)


@router.post("/{model_id}/activate", response_model=ActivateResponse)
def activate(
    model_id: int, admin: AdminUser, db: Db, store: Store, request: Request
) -> ActivateResponse:
    """Promote a version: the previous active model is archived, an admin threshold override
    is reset to the new model's own threshold, and the API scores with it immediately (no
    restart). Audited. Fails with 409 (and changes nothing) if the artifacts cannot be loaded."""
    row = get_row(db, model_id)
    try:
        row, reset = promote(db, row, admin, store, request)
    except ModelLoadError as exc:
        db.rollback()
        raise HTTPException(status.HTTP_409_CONFLICT, str(exc)) from exc
    db.refresh(row)
    return ActivateResponse(model=out(row, store), threshold_override_reset=reset)


@router.post("/{model_id}/archive", response_model=ModelOut)
def archive(model_id: int, admin: AdminUser, db: Db, store: Store, request: Request) -> ModelOut:
    """Retire a candidate (audited). The active model cannot be archived (409)."""
    row = get_row(db, model_id)
    try:
        archive_version(db, row, admin, request)
    except RegistryError as exc:
        db.rollback()
        raise HTTPException(status.HTTP_409_CONFLICT, str(exc)) from exc
    db.commit()
    return out(row, store)
