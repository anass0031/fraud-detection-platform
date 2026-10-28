"""Scoring service: one code path for `POST /score` and the batch script.

raw transaction -> features (shared fraudml online path, history from PostgreSQL)
-> LightGBM -> isotonic calibration -> expected loss + alert rule -> Isolation Forest flag
-> SHAP top factors -> `predictions` row -> review-queue case if alerted or flagged.
"""

from __future__ import annotations

import datetime as dt
import math
import time
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from fraudml import config
from fraudml.explain.explain import TOP_K, explain_transaction
from fraudml.features.keys import make_uid
from fraudml.features.pipeline import RAW_INPUT_COLUMNS, compute_features_online_frame
from fraudml.models.calibration import expected_loss
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.db.models import Prediction, Transaction
from app.services.cases import open_case
from app.services.history import PostgresHistory
from app.services.model_store import ScoringBundle
from app.services.settings import DEFAULT_RISK_BANDS, AlertPolicy, alert_policy
from app.services.transactions_loader import READABLE

# Risk level from the calibrated probability: three upper bounds (exclusive) -> four levels.
# The bounds come from the `thresholds` setting (defaults 5% / 20% / 50%).
RISK_LEVELS = ("low", "medium", "high", "critical")


class TransactionConflict(ValueError):
    """The TransactionID is already stored with different data."""


def risk_level(probability: float, bands: Sequence[float] = DEFAULT_RISK_BANDS) -> str:
    for bound, level in zip(bands, RISK_LEVELS, strict=False):
        if probability < bound:
            return level
    return RISK_LEVELS[-1]


def _json_value(v: Any) -> Any:
    if v is None or (isinstance(v, float | np.floating) and math.isnan(v)):
        return None
    if isinstance(v, bool | np.bool_):
        return bool(v)
    if isinstance(v, int | np.integer):
        return int(v)
    if isinstance(v, float | np.floating):
        return float(v)
    return str(v)


def features_to_json(X: pd.DataFrame) -> dict[str, Any]:
    row = X.iloc[0]
    return {c: _json_value(row[c]) for c in X.columns}


@dataclass
class ScoreOutcome:
    transaction: Transaction
    prediction: Prediction
    already_scored: bool


def transaction_from_raw(raw: dict[str, Any], source: str) -> Transaction:
    """A new Transaction row from raw IEEE-CIS fields (no ground truth: it is unknown)."""
    frame = pd.DataFrame([{c: raw.get(c) for c in RAW_INPUT_COLUMNS}])
    uid = make_uid(frame).iloc[0]
    ref = pd.Timestamp(config.REFERENCE_DATETIME)
    return Transaction(
        external_id=int(raw[config.ID_COL]),
        transaction_dt=int(raw[config.TIME_COL]),
        transaction_time=(ref + pd.Timedelta(seconds=int(raw[config.TIME_COL]))).to_pydatetime(),
        amount=raw["TransactionAmt"],
        **{col: raw.get(src) for col, src in READABLE.items()},
        customer_key=None if pd.isna(uid) else str(uid),
        raw={c: raw.get(c) for c in RAW_INPUT_COLUMNS},
        source=source,
    )


@dataclass
class Scored:
    """Everything the service computes for one transaction (nothing written yet)."""

    features: pd.DataFrame  # one row, encoded (model input)
    fraud_score: float
    expected_loss: float
    is_alert: bool
    anomaly_score: float | None
    is_unusual: bool
    risk_level: str
    explanation: dict[str, Any]
    latency_ms: float


def compute_score(
    db: Session, bundle: ScoringBundle, raw: dict[str, Any], policy: AlertPolicy | None = None
) -> Scored:
    """Features (history from PostgreSQL) -> calibrated score -> policy -> anomaly -> SHAP.

    `policy` defaults to the effective one (admin settings, else the model's threshold).
    Read-only: the transaction itself must not be in its own history (strictly earlier rows
    only, so a stored transaction is excluded automatically).
    """
    t0 = time.perf_counter()
    X = compute_features_online_frame(raw, PostgresHistory(db), bundle.encoder)
    model = bundle.model
    if policy is None:
        policy = alert_policy(db, model.metadata["threshold"])
    Xm = X[model.features]
    amount = float(raw["TransactionAmt"])
    prob = float(model.predict_proba(Xm)[0])
    exp_loss = float(expected_loss(prob, amount))
    # Same rule as the model's own is_alert, with the effective threshold.
    is_alert = bool(model.alert_score(prob, amount) >= policy.threshold)

    anomaly_score, is_unusual = None, False
    if bundle.anomaly is not None:
        anomaly_score = float(bundle.anomaly.score(X)[0])
        # Safety net: only flags transactions the main model did not alert on.
        is_unusual = (not is_alert) and anomaly_score >= bundle.anomaly.cutoff

    explanation = explain_transaction(model.booster, Xm, k=TOP_K, calibrator=model.calibrator)
    return Scored(
        features=X,
        fraud_score=prob,
        expected_loss=exp_loss,
        is_alert=is_alert,
        anomaly_score=anomaly_score,
        is_unusual=bool(is_unusual),
        risk_level=risk_level(prob, policy.risk_bands),
        explanation=explanation,
        latency_ms=(time.perf_counter() - t0) * 1000,
    )


def score_stored(db: Session, bundle: ScoringBundle, txn: Transaction) -> Prediction:
    """Score a stored transaction with the bundle and add the prediction to the session.

    Opens a review-queue case (same DB transaction) if the prediction is an alert or an
    "unusual pattern" flag.
    """
    s = compute_score(db, bundle, txn.raw)
    txn.features = features_to_json(s.features)
    pred = Prediction(
        transaction=txn,
        model_version_id=bundle.model_version_id,
        fraud_score=s.fraud_score,
        expected_loss=s.expected_loss,
        is_alert=s.is_alert,
        anomaly_score=s.anomaly_score,
        is_unusual=s.is_unusual,
        risk_level=s.risk_level,
        shap_values=s.explanation,
        base_value=s.explanation["base_value"],
        latency_ms=s.latency_ms,
        created_at=dt.datetime.now(dt.UTC),
    )
    db.add(pred)
    db.flush()
    open_case(db, pred)
    return pred


def score_raw(
    db: Session, bundle: ScoringBundle, raw: dict[str, Any], source: str = "realtime"
) -> ScoreOutcome:
    """Store the transaction if new, then score it (or return its existing prediction)."""
    external_id = int(raw[config.ID_COL])
    txn = db.scalar(select(Transaction).where(Transaction.external_id == external_id))
    if txn is None:
        txn = transaction_from_raw(raw, source)
        db.add(txn)
        db.flush()
    else:
        sent = {c: raw.get(c) for c in RAW_INPUT_COLUMNS}
        if sent != {c: txn.raw.get(c) for c in RAW_INPUT_COLUMNS}:
            raise TransactionConflict(
                f"TransactionID {external_id} is already stored with different data"
            )
        existing = db.scalar(
            select(Prediction).where(
                Prediction.transaction_id == txn.id,
                Prediction.model_version_id == bundle.model_version_id,
            )
        )
        if existing is not None:
            return ScoreOutcome(txn, existing, already_scored=True)
    return ScoreOutcome(txn, score_stored(db, bundle, txn), already_scored=False)
