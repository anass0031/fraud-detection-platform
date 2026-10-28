import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI
from sqlalchemy.exc import SQLAlchemyError

from app.api.v1 import auth, cases, health, monitoring, reports, score, users
from app.api.v1 import settings as settings_routes
from app.core.config import get_settings
from app.db.session import SessionLocal
from app.services.model_store import store

settings = get_settings()
log = logging.getLogger("uvicorn.error")


@asynccontextmanager
async def lifespan(_app: FastAPI) -> AsyncIterator[None]:
    """Load the active scoring bundle once at startup (the API still starts without one)."""
    try:
        with SessionLocal() as db:
            bundle = store.load_active(db)
    except SQLAlchemyError as exc:
        bundle, store.error = None, f"database unavailable at startup: {exc.__class__.__name__}"
    if bundle:
        log.info("Loaded model %s (anomaly %s)", bundle.version, bundle.anomaly_version)
    else:
        log.warning("No scoring model loaded: %s", store.error)
    yield


app = FastAPI(title=settings.app_name, version=settings.version, lifespan=lifespan)
app.include_router(health.router, prefix=settings.api_v1_prefix)
app.include_router(auth.router, prefix=settings.api_v1_prefix)
app.include_router(users.router, prefix=settings.api_v1_prefix)
app.include_router(score.router, prefix=settings.api_v1_prefix)
app.include_router(cases.router, prefix=settings.api_v1_prefix)
app.include_router(settings_routes.router, prefix=settings.api_v1_prefix)
app.include_router(reports.router, prefix=settings.api_v1_prefix)
app.include_router(monitoring.router, prefix=settings.api_v1_prefix)
