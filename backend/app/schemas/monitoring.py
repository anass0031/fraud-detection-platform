"""Monitoring responses. Aggregates only: never a per-transaction label."""

from __future__ import annotations

import datetime as dt

from pydantic import BaseModel, Field

Severity = str  # "stable" | "moderate" | "significant"


class PerformanceRow(BaseModel):
    day: dt.date
    n: int
    n_fraud: int
    alerts: int
    tp: int
    fp: int
    fn: int
    precision: float | None = Field(description="Share of alerts that were fraud")
    recall: float | None = Field(description="Share of frauds that were alerted")
    pr_auc: float | None
    precision_at_capacity: float | None = Field(description="Precision of the day's top-k")
    fraud_value_caught: float | None = Field(description="Share of fraud $ inside the alerts")
    total_cost: float
    no_model_cost: float
    net_savings: float


class PerformanceTotals(BaseModel):
    first_day: dt.date
    last_day: dt.date
    days: int
    n: int
    n_fraud: int
    alerts: int
    alerts_per_day: float
    precision: float | None
    recall: float | None
    pr_auc: float | None = Field(description="Recomputed on all labelled rows of the period")
    fraud_value_caught: float | None
    total_cost: float
    no_model_cost: float
    net_savings: float


class PerformanceResponse(BaseModel):
    model_version: str | None
    days: list[PerformanceRow]
    total: PerformanceTotals | None = Field(description="The whole requested period")
    last_7_days: PerformanceTotals | None = Field(description="The last 7 days of the period")


class DriftCell(BaseModel):
    day: dt.date
    psi: float
    severity: Severity


class FeatureDrift(BaseModel):
    feature: str
    max_psi: float
    mean_psi: float
    latest_psi: float
    latest_severity: Severity
    days_significant: int
    cells: list[DriftCell]


class ScoreDrift(BaseModel):
    day: dt.date
    psi: float
    ks_stat: float | None
    severity: Severity


class DriftResponse(BaseModel):
    model_version: str | None
    days: list[dt.date]
    features: list[FeatureDrift] = Field(description="Sorted by max PSI, highest first")
    score: list[ScoreDrift] = Field(description="Drift of the model's score distribution")
    thresholds: dict[str, float] = Field(description="PSI bands: moderate / significant")


class AlertDay(BaseModel):
    day: dt.date = Field(description="Transaction day")
    model_alerts: int
    unusual: int
    open: int = Field(description="Cases of that day still open now")
    closed: int


class AlertsResponse(BaseModel):
    days: list[AlertDay]
    backlog: dict[str, int] = Field(description="Open cases now, by status")
    capacity_per_day: int


class FeedbackResponse(BaseModel):
    decisions: dict[str, int] = Field(description="Decisions recorded, by type")
    cases_closed: int
    confirmed_share: float | None = Field(
        description="Closed cases confirmed as fraud: how often analysts agree with the model"
    )
    checked_against_labels: int = Field(description="Closed cases whose label is revealed")
    matching_labels: int
    mean_hours_to_decision: float | None = Field(
        description="Case opened -> closed (not meaningful for replayed days: opened in bulk)"
    )
