"""Model registry responses (admin)."""

from __future__ import annotations

import datetime as dt
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


class ModelOut(BaseModel):
    model_config = ConfigDict(from_attributes=True, protected_namespaces=())

    id: int
    version: str
    algorithm: str
    status: Literal["candidate", "active", "archived"]
    anomaly_version: str | None
    policy: str
    threshold: float = Field(description="The model's own alert threshold")
    calibration_method: str | None
    trained_at: dt.datetime | None
    created_at: dt.datetime
    training_data_window: dict[str, Any] | None
    metrics: dict[str, Any] | None = Field(
        description="validation metrics, anomaly validation, and (candidates from the retrain "
        "script) the champion vs challenger comparison"
    )
    hyperparams: dict[str, Any] | None
    notes: str | None
    serving: bool = Field(False, description="This version is the one the API is scoring with")


class ActivateResponse(BaseModel):
    model: ModelOut
    threshold_override_reset: bool = Field(
        description="An admin threshold override existed and was reset to the model's own value"
    )
