"""Review queue schemas. Ground truth (`isFraud`) is never part of any of them."""

from __future__ import annotations

import datetime as dt
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from app.schemas.score import RiskFactor

CaseStatus = Literal["new", "assigned", "escalated", "closed"]
CaseReason = Literal["model_score", "anomaly"]
RiskLevel = Literal["low", "medium", "high", "critical"]
DecisionKind = Literal["confirmed_fraud", "false_positive", "escalate"]
FinalDecision = Literal["confirmed_fraud", "false_positive"]
CaseSort = Literal["priority", "amount", "transaction_time", "created_at"]


class UserRef(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    full_name: str
    email: str


class CaseListItem(BaseModel):
    id: int
    transaction_id: int = Field(description="TransactionID")
    transaction_time: dt.datetime
    amount: float
    product_code: str | None
    card_network: str | None
    fraud_score: float = Field(description="Calibrated fraud probability")
    expected_loss: float = Field(description="fraud_score x amount (USD); priority inside a tier")
    risk_level: RiskLevel
    reason: CaseReason
    priority_tier: int = Field(description="1 = model alert, 2 = unusual pattern")
    status: CaseStatus
    assignee: UserRef | None
    final_decision: FinalDecision | None
    created_at: dt.datetime
    updated_at: dt.datetime
    closed_at: dt.datetime | None


class CasePage(BaseModel):
    items: list[CaseListItem]
    total: int = Field(description="Number of cases matching the filters")
    limit: int
    offset: int


class TransactionFacts(BaseModel):
    transaction_id: int
    transaction_time: dt.datetime
    amount: float
    product_code: str | None
    card_network: str | None
    card_type: str | None
    payer_email_domain: str | None
    device_type: str | None


class PredictionOut(BaseModel):
    model_version: str
    fraud_score: float
    expected_loss: float
    risk_level: RiskLevel
    is_alert: bool
    is_unusual: bool
    anomaly_score: float | None
    base_value: float | None
    top_factors: list[RiskFactor]
    scored_at: dt.datetime


class HistoryItem(BaseModel):
    """An earlier transaction of the same card/customer (no labels, no decisions)."""

    transaction_id: int
    transaction_time: dt.datetime
    amount: float
    product_code: str | None
    had_case: bool


class DecisionOut(BaseModel):
    id: int
    decision: DecisionKind
    comment: str | None
    analyst: UserRef
    created_at: dt.datetime


class CaseDetail(CaseListItem):
    transaction: TransactionFacts
    prediction: PredictionOut
    card_history: list[HistoryItem] = Field(description="Up to 10 earlier transactions")
    decisions: list[DecisionOut]


class AssignRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    user_id: int = Field(description="Active user who receives the case")


class DecisionRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    decision: DecisionKind
    comment: str | None = Field(
        None, max_length=2000, description="Required to escalate; optional otherwise"
    )

    @field_validator("comment")
    @classmethod
    def blank_is_none(cls, v: str | None) -> str | None:
        return (v or "").strip() or None

    @model_validator(mode="after")
    def escalation_needs_a_comment(self) -> DecisionRequest:
        if self.decision == "escalate" and not self.comment:
            raise ValueError("a comment is required to escalate")
        return self


class TimelineEntry(BaseModel):
    """One audit event of a case (case.create, case.claim, case.assign, case.decide)."""

    id: int
    action: str
    actor: UserRef | None = Field(description="None = system (scoring / scripts)")
    before: dict[str, Any] | None
    after: dict[str, Any] | None
    comment: str | None = Field(description="Decision comment (case.decide only)")
    created_at: dt.datetime
