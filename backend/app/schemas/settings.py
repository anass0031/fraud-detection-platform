"""Threshold and cost settings (admin only)."""

from __future__ import annotations

import datetime as dt

from pydantic import BaseModel, ConfigDict, Field, field_validator

from app.schemas.cases import UserRef


class ThresholdsIn(BaseModel):
    model_config = ConfigDict(extra="forbid")

    alert_threshold: float | None = Field(
        gt=0,
        le=100_000,
        description="Alert when the model's alert score (expected loss, USD) >= this value; "
        "null = use the active model's threshold",
    )
    risk_bands: list[float] = Field(
        min_length=3,
        max_length=3,
        description="Upper bounds of low / medium / high on the calibrated probability",
    )

    @field_validator("risk_bands")
    @classmethod
    def increasing_probabilities(cls, v: list[float]) -> list[float]:
        if not all(0 < b < 1 for b in v):
            raise ValueError("risk bands must be between 0 and 1")
        if not all(a < b for a, b in zip(v, v[1:], strict=False)):
            raise ValueError("risk bands must be strictly increasing")
        return v


class ThresholdsOut(ThresholdsIn):
    model_config = ConfigDict(extra="ignore")

    alert_threshold: float | None
    risk_bands: list[float]
    effective_threshold: float | None = Field(description="Threshold in force now")
    model_version: str | None = Field(description="Active model")
    model_threshold: float | None = Field(description="The active model's own threshold")
    alert_policy: str | None = Field(description="What the threshold applies to")
    updated_by: UserRef | None
    updated_at: dt.datetime | None


class CostsIn(BaseModel):
    model_config = ConfigDict(extra="forbid")

    review_cost: float = Field(ge=0, le=10_000, description="USD per reviewed alert")
    friction_cost: float = Field(ge=0, le=10_000, description="USD per false alarm")
    capacity_per_day: int = Field(ge=1, le=10_000, description="Cases analysts can review a day")


class CostsOut(CostsIn):
    model_config = ConfigDict(extra="ignore")

    updated_by: UserRef | None
    updated_at: dt.datetime | None
