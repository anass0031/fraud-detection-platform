"""Monitoring (any logged-in user, read-only): performance on replayed days, drift, alert
volume and backlog, analyst feedback. Aggregates only; labels are never returned per case."""

from __future__ import annotations

import datetime as dt
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy.orm import Session

from app.api.deps import CurrentUser
from app.db.session import get_db
from app.schemas.monitoring import (
    AlertsResponse,
    DriftResponse,
    FeedbackResponse,
    PerformanceResponse,
)
from app.services import monitoring_read as svc

router = APIRouter(prefix="/monitoring", tags=["monitoring"])

Db = Annotated[Session, Depends(get_db)]
From = Annotated[dt.date | None, Query(alias="from", description="First day, inclusive")]
To = Annotated[dt.date | None, Query(alias="to", description="Last day, inclusive")]


def _check(start: dt.date | None, end: dt.date | None) -> None:
    if start and end and end < start:
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_CONTENT, "'to' is before 'from'")


@router.get("/performance", response_model=PerformanceResponse)
def performance(_user: CurrentUser, db: Db, start: From = None, end: To = None) -> dict:
    """Daily performance of the active model on days whose labels are revealed, with exact
    totals for the period and for its last 7 days (PR-AUC recomputed, not averaged)."""
    _check(start, end)
    model = svc.active_model(db)
    if model is None:
        return {"model_version": None, "days": [], "total": None, "last_7_days": None}
    snaps = svc.snapshots(db, model.id, start, end)
    return {
        "model_version": model.version,
        "days": [svc.row(s) for s in snaps],
        "total": svc.totals(db, model.id, snaps),
        "last_7_days": svc.totals(db, model.id, snaps[-7:]),
    }


@router.get("/drift", response_model=DriftResponse)
def drift(_user: CurrentUser, db: Db, start: From = None, end: To = None) -> dict:
    """PSI per feature and day against the training reference (features sorted by their
    highest PSI), and the drift of the model's score."""
    _check(start, end)
    model = svc.active_model(db)
    if model is None:
        return {"model_version": None, **svc.drift(db, -1, start, end)}
    return {"model_version": model.version, **svc.drift(db, model.id, start, end)}


@router.get("/alerts", response_model=AlertsResponse)
def alerts(_user: CurrentUser, db: Db, start: From = None, end: To = None) -> dict:
    """Cases opened per transaction day (model alerts, unusual patterns) and the open backlog."""
    _check(start, end)
    return svc.alerts(db, start, end)


@router.get("/feedback", response_model=FeedbackResponse)
def feedback(_user: CurrentUser, db: Db) -> dict:
    """Analyst decisions and how often they agree with the model (and, once revealed, with
    the labels: counts only)."""
    return svc.feedback(db)
