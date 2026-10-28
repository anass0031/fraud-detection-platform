"""Admin-editable settings stored in the `settings` table (a missing key = defaults).

- `thresholds`: alert threshold override (None = the active model's own threshold) and the
  three risk-band bounds on the calibrated probability (low | medium | high | critical).
- `costs`: review cost, customer-friction cost and daily review capacity (business cost model;
  used by monitoring and the threshold page).

Changes apply to transactions scored afterwards, never retroactively. Every update writes a
`settings.update` audit row with the full before / after values.
"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass
from typing import Any

from fastapi import Request
from sqlalchemy.orm import Session

from app.db.models import Setting, User
from app.services import audit

THRESHOLDS = "thresholds"
COSTS = "costs"

DEFAULT_RISK_BANDS = (0.05, 0.20, 0.50)
DEFAULTS: dict[str, dict[str, Any]] = {
    THRESHOLDS: {"alert_threshold": None, "risk_bands": list(DEFAULT_RISK_BANDS)},
    COSTS: {"review_cost": 5.0, "friction_cost": 2.0, "capacity_per_day": 150},
}


def get_setting(db: Session, key: str) -> tuple[dict[str, Any], Setting | None]:
    """(value with defaults filled in, stored row or None)."""
    row = db.get(Setting, key)
    value = {**DEFAULTS[key], **(row.value if row is not None else {})}
    return value, row


def update_setting(
    db: Session, key: str, value: dict[str, Any], admin: User, request: Request | None = None
) -> Setting:
    """Replace a setting (already validated) and audit it, in the caller's transaction."""
    before, row = get_setting(db, key)
    if row is None:
        row = Setting(key=key, value=value)
        db.add(row)
    else:
        row.value = value
    row.updated_by = admin.id
    row.updated_at = dt.datetime.now(dt.UTC)
    db.flush()
    audit.record(
        db,
        "settings.update",
        actor=admin,
        entity_type="setting",
        entity_id=key,
        before=before,
        after=value,
        request=request,
    )
    return row


@dataclass(frozen=True)
class AlertPolicy:
    """What scoring needs: the threshold applied to the model's alert score, and risk bands."""

    threshold: float
    risk_bands: tuple[float, ...] = DEFAULT_RISK_BANDS
    overridden: bool = False


def model_policy(model_threshold: float) -> AlertPolicy:
    """The model's own policy with default bands (ignores settings; used by the skew check)."""
    return AlertPolicy(threshold=float(model_threshold))


def alert_policy(db: Session, model_threshold: float) -> AlertPolicy:
    """Effective policy: the admin's threshold override if set, else the model's threshold."""
    value, _ = get_setting(db, THRESHOLDS)
    override = value["alert_threshold"]
    return AlertPolicy(
        threshold=float(model_threshold if override is None else override),
        risk_bands=tuple(float(b) for b in value["risk_bands"]),
        overridden=override is not None,
    )
