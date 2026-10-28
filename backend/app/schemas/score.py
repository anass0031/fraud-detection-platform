"""Request / response schemas of `POST /api/v1/score`.

The request uses the raw IEEE-CIS field names (the inputs of the shared feature pipeline).
Only TransactionID, TransactionDT and TransactionAmt are required; missing fields are treated
as missing values, exactly like NaN in training. Unknown fields (e.g. isFraud) are rejected.
"""

from __future__ import annotations

import datetime as dt
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

Num = float | None
Str = str | None


class TransactionIn(BaseModel):
    model_config = ConfigDict(extra="forbid")

    TransactionID: int = Field(ge=0, description="Unique transaction id")
    TransactionDT: int = Field(ge=0, description="Seconds since 2017-12-01 00:00 (dataset clock)")
    TransactionAmt: float = Field(gt=0, description="Amount in USD")
    ProductCD: Str = Field(None, max_length=8)
    card1: Num = None
    card2: Num = None
    card4: Str = Field(None, max_length=32, description="Card network, e.g. visa")
    card6: Str = Field(None, max_length=32, description="Card type, e.g. debit")
    addr1: Num = None
    dist1: Num = None
    P_emaildomain: Str = Field(None, max_length=64, description="Payer e-mail domain")
    R_emaildomain: Str = Field(None, max_length=64, description="Recipient e-mail domain")
    DeviceType: Str = Field(None, max_length=16)
    DeviceInfo: Str = Field(None, max_length=128)
    id_01: Num = None
    id_30: Str = Field(None, max_length=64, description="Operating system")
    id_31: Str = Field(None, max_length=64, description="Browser")
    D1: Num = None
    D4: Num = None
    D10: Num = None
    D15: Num = None
    C1: Num = None
    C2: Num = None
    C3: Num = None
    C4: Num = None
    C5: Num = None
    C6: Num = None
    C7: Num = None
    C8: Num = None
    C9: Num = None
    C10: Num = None
    C11: Num = None
    C12: Num = None
    C13: Num = None
    C14: Num = None
    M1: Str = Field(None, max_length=8)
    M2: Str = Field(None, max_length=8)
    M3: Str = Field(None, max_length=8)
    M4: Str = Field(None, max_length=8)
    M5: Str = Field(None, max_length=8)
    M6: Str = Field(None, max_length=8)
    M7: Str = Field(None, max_length=8)
    M8: Str = Field(None, max_length=8)
    M9: Str = Field(None, max_length=8)


class RiskFactor(BaseModel):
    key: str
    masked: bool = Field(description="Anonymised feature family (list after readable factors)")
    label: str
    value: Any = None
    contribution: float = Field(description="SHAP contribution in log-odds")
    direction: str
    strength: str
    text: str = Field(description="Plain-language sentence for analysts")


class ScoreResponse(BaseModel):
    transaction_id: int = Field(description="TransactionID")
    model_version: str
    anomaly_version: str | None
    fraud_score: float = Field(description="Calibrated fraud probability")
    expected_loss: float = Field(description="fraud_score x amount, in USD")
    is_alert: bool
    alert_policy: str
    alert_threshold: float
    risk_level: Literal["low", "medium", "high", "critical"]
    is_unusual: bool = Field(description="Unusual-pattern flag (anomaly safety net, no alert)")
    anomaly_score: float | None
    base_value: float | None
    top_factors: list[RiskFactor]
    latency_ms: float | None
    scored_at: dt.datetime
    already_scored: bool = Field(description="True when an earlier prediction was returned")
