"""Champion / challenger promotion: activate a registered model without restarting the API.

Order matters, so a failure never leaves the API without a working model:
1. load the new version's artifacts (fails -> nothing changes);
2. switch the active row (old -> archived), audited `model.activate`;
3. reset the alert-threshold override (it was tuned for the previous model), audited;
4. commit; 5. swap the in-memory bundle.
"""

from __future__ import annotations

from fastapi import Request
from sqlalchemy.orm import Session

from app.db.models import ModelVersion, User
from app.services import settings as settings_svc
from app.services.model_store import ModelStore, activate_version


class ModelLoadError(RuntimeError):
    """The version's artifacts could not be loaded (missing, incompatible)."""


def promote(
    db: Session, row: ModelVersion, admin: User, store: ModelStore, request: Request | None = None
) -> tuple[ModelVersion, bool]:
    """Make `row` the active model. Returns (row, threshold_override_was_reset)."""
    if row.status == "active" and store.bundle is not None and store.bundle.version == row.version:
        return row, False
    try:
        bundle = store.prepare(row)
    except Exception as exc:  # any loader error (LightGBM, joblib, files): nothing changes
        raise ModelLoadError(f"{row.version} cannot be loaded: {exc}") from exc

    activate_version(db, row.version, actor=admin, request=request)

    reset = False
    value, _ = settings_svc.get_setting(db, settings_svc.THRESHOLDS)
    if value["alert_threshold"] is not None:
        settings_svc.update_setting(
            db,
            settings_svc.THRESHOLDS,
            {**value, "alert_threshold": None},
            admin,
            request,
        )
        reset = True

    db.commit()
    store.swap(bundle)
    return row, reset
