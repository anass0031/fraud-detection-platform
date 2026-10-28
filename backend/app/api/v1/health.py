import datetime as dt
from typing import Annotated, Literal

from fastapi import APIRouter, Depends, Response, status
from pydantic import BaseModel
from sqlalchemy import text
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session

from app.core.config import get_settings
from app.db.session import get_db
from app.services.model_store import ModelStore, get_model_store

router = APIRouter(tags=["health"])


class ModelStatus(BaseModel):
    status: Literal["loaded", "not_loaded"]
    version: str | None = None
    anomaly_version: str | None = None
    loaded_at: dt.datetime | None = None
    error: str | None = None


class HealthResponse(BaseModel):
    status: Literal["ok", "degraded"]
    database: Literal["ok", "error"]
    version: str
    model: ModelStatus


@router.get("/health", response_model=HealthResponse)
def health(
    response: Response,
    db: Annotated[Session, Depends(get_db)],
    models: Annotated[ModelStore, Depends(get_model_store)],
) -> HealthResponse:
    """Liveness check. 503 when the database is unreachable; "degraded" when no model is loaded."""
    try:
        db.execute(text("SELECT 1"))
        database = "ok"
    except SQLAlchemyError:
        database = "error"
        response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE

    bundle = models.bundle
    model = (
        ModelStatus(
            status="loaded",
            version=bundle.version,
            anomaly_version=bundle.anomaly_version,
            loaded_at=bundle.loaded_at,
        )
        if bundle
        else ModelStatus(status="not_loaded", error=models.error)
    )
    return HealthResponse(
        status="ok" if database == "ok" and bundle else "degraded",
        database=database,
        version=get_settings().version,
        model=model,
    )
