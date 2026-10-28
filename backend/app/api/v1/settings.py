"""Threshold and cost settings (admin only; every change is audited)."""

from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends, Request
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.api.deps import require_role
from app.db.models import ModelVersion, Setting, User
from app.db.session import get_db
from app.schemas.cases import UserRef
from app.schemas.settings import CostsIn, CostsOut, ThresholdsIn, ThresholdsOut
from app.services import settings as svc

router = APIRouter(prefix="/settings", tags=["settings"])

AdminUser = Annotated[User, Depends(require_role("admin"))]
Db = Annotated[Session, Depends(get_db)]


def _meta(db: Session, row: Setting | None) -> dict:
    if row is None:
        return {"updated_by": None, "updated_at": None}
    user = db.get(User, row.updated_by) if row.updated_by else None
    return {
        "updated_by": UserRef.model_validate(user) if user else None,
        "updated_at": row.updated_at,
    }


def _thresholds(db: Session) -> ThresholdsOut:
    value, row = svc.get_setting(db, svc.THRESHOLDS)
    model = db.scalar(select(ModelVersion).where(ModelVersion.status == "active"))
    override = value["alert_threshold"]
    return ThresholdsOut(
        **value,
        effective_threshold=override if override is not None else (model and model.threshold),
        model_version=model.version if model else None,
        model_threshold=model.threshold if model else None,
        alert_policy=model.policy if model else None,
        **_meta(db, row),
    )


def _costs(db: Session) -> CostsOut:
    value, row = svc.get_setting(db, svc.COSTS)
    return CostsOut(**value, **_meta(db, row))


@router.get("/thresholds", response_model=ThresholdsOut)
def get_thresholds(_admin: AdminUser, db: Db) -> ThresholdsOut:
    """Alert threshold (override or the active model's) and risk-level bands."""
    return _thresholds(db)


@router.put("/thresholds", response_model=ThresholdsOut)
def put_thresholds(body: ThresholdsIn, admin: AdminUser, db: Db, request: Request) -> ThresholdsOut:
    """Change the alert threshold (null = back to the model's) and the risk bands.
    Applies to transactions scored from now on; existing cases are not changed."""
    svc.update_setting(db, svc.THRESHOLDS, body.model_dump(), admin, request)
    db.commit()
    return _thresholds(db)


@router.get("/costs", response_model=CostsOut)
def get_costs(_admin: AdminUser, db: Db) -> CostsOut:
    """Business cost assumptions: review cost, false-alarm friction, daily capacity."""
    return _costs(db)


@router.put("/costs", response_model=CostsOut)
def put_costs(body: CostsIn, admin: AdminUser, db: Db, request: Request) -> CostsOut:
    svc.update_setting(db, svc.COSTS, body.model_dump(), admin, request)
    db.commit()
    return _costs(db)
