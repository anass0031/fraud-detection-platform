"""Database tables: transactions, model registry, predictions, users, refresh tokens, audit log,
review queue (cases, decisions), settings and monitoring (daily metrics, drift).
"""

from __future__ import annotations

import datetime as dt
from typing import Any

from sqlalchemy import (
    BigInteger,
    Boolean,
    CheckConstraint,
    Date,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    MetaData,
    Numeric,
    String,
    Text,
    UniqueConstraint,
    func,
    text,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship

# Stable constraint names, so Alembic migrations stay readable and reversible.
NAMING_CONVENTION = {
    "ix": "ix_%(column_0_label)s",
    "uq": "uq_%(table_name)s_%(column_0_name)s",
    "ck": "ck_%(table_name)s_%(constraint_name)s",
    "fk": "fk_%(table_name)s_%(column_0_name)s_%(referred_table_name)s",
    "pk": "pk_%(table_name)s",
}

TRANSACTION_SOURCES = ("history", "batch", "realtime", "replay")
MODEL_STATUSES = ("candidate", "active", "archived")
RISK_LEVELS = ("low", "medium", "high", "critical")
USER_ROLES = ("analyst", "admin")
CASE_REASONS = ("model_score", "anomaly")
CASE_STATUSES = ("new", "assigned", "escalated", "closed")
DECISIONS = ("confirmed_fraud", "false_positive", "escalate")
DRIFT_SEVERITIES = ("stable", "moderate", "significant")


class Base(DeclarativeBase):
    metadata = MetaData(naming_convention=NAMING_CONVENTION)


class Transaction(Base):
    """One card transaction. `raw` keeps every input column the feature pipeline needs."""

    __tablename__ = "transactions"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True)
    external_id: Mapped[int] = mapped_column(BigInteger, unique=True)  # IEEE-CIS TransactionID
    transaction_dt: Mapped[int] = mapped_column(BigInteger)  # seconds offset (TransactionDT)
    transaction_time: Mapped[dt.datetime] = mapped_column(DateTime(timezone=False))
    amount: Mapped[float] = mapped_column(Numeric(12, 3))
    product_code: Mapped[str | None] = mapped_column(String(8))
    card_network: Mapped[str | None] = mapped_column(String(32))
    card_type: Mapped[str | None] = mapped_column(String(32))
    payer_email_domain: Mapped[str | None] = mapped_column(String(64))
    device_type: Mapped[str | None] = mapped_column(String(16))
    # Pseudo customer id (card + address + account-start day), used for history lookups only.
    customer_key: Mapped[str | None] = mapped_column(String(128))
    raw: Mapped[dict[str, Any]] = mapped_column(JSONB)
    features: Mapped[dict[str, Any] | None] = mapped_column(JSONB)
    # Ground truth: evaluation only, never returned to analysts or by the scoring API.
    ground_truth: Mapped[bool | None] = mapped_column(Boolean)
    ground_truth_revealed_at: Mapped[dt.datetime | None] = mapped_column(DateTime(timezone=True))
    source: Mapped[str] = mapped_column(String(16))
    created_at: Mapped[dt.datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )

    predictions: Mapped[list[Prediction]] = relationship(back_populates="transaction")

    __table_args__ = (
        Index("ix_transactions_customer_key_dt", "customer_key", "transaction_dt"),
        Index("ix_transactions_transaction_time", "transaction_time"),
        Index("ix_transactions_source", "source"),
        CheckConstraint(f"source IN {TRANSACTION_SOURCES}", name="source"),
    )


class ModelVersion(Base):
    """A scoring bundle: LightGBM model + calibrator + alert policy, and its anomaly detector."""

    __tablename__ = "model_versions"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    version: Mapped[str] = mapped_column(String(64), unique=True)
    algorithm: Mapped[str] = mapped_column(String(32))
    artifact_path: Mapped[str] = mapped_column(String(255))
    anomaly_version: Mapped[str | None] = mapped_column(String(64))
    feature_list: Mapped[list[str]] = mapped_column(JSONB)
    hyperparams: Mapped[dict[str, Any] | None] = mapped_column(JSONB)
    metrics: Mapped[dict[str, Any] | None] = mapped_column(JSONB)
    calibration_method: Mapped[str | None] = mapped_column(String(32))
    policy: Mapped[str] = mapped_column(String(32))
    threshold: Mapped[float] = mapped_column(Float)
    status: Mapped[str] = mapped_column(String(16), server_default="candidate")
    trained_at: Mapped[dt.datetime | None] = mapped_column(DateTime(timezone=True))
    training_data_window: Mapped[dict[str, Any] | None] = mapped_column(JSONB)
    notes: Mapped[str | None] = mapped_column(Text)
    created_at: Mapped[dt.datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )

    __table_args__ = (
        # At most one active scoring bundle at any time.
        Index(
            "uq_model_versions_one_active",
            "status",
            unique=True,
            postgresql_where=text("status = 'active'"),
        ),
        CheckConstraint(f"status IN {MODEL_STATUSES}", name="status"),
    )


class Prediction(Base):
    """The output of one model version for one transaction."""

    __tablename__ = "predictions"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True)
    transaction_id: Mapped[int] = mapped_column(
        ForeignKey("transactions.id", ondelete="CASCADE"), index=True
    )
    model_version_id: Mapped[int] = mapped_column(ForeignKey("model_versions.id"), index=True)
    fraud_score: Mapped[float] = mapped_column(Float)  # calibrated probability
    expected_loss: Mapped[float] = mapped_column(Float)
    is_alert: Mapped[bool] = mapped_column(Boolean)
    anomaly_score: Mapped[float | None] = mapped_column(Float)
    is_unusual: Mapped[bool] = mapped_column(Boolean, server_default="false")
    risk_level: Mapped[str] = mapped_column(String(16))
    # Top contributions (grouped factors + base value) as returned by fraudml.explain.
    shap_values: Mapped[dict[str, Any]] = mapped_column(JSONB)
    base_value: Mapped[float | None] = mapped_column(Float)
    latency_ms: Mapped[float | None] = mapped_column(Float)
    created_at: Mapped[dt.datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )

    transaction: Mapped[Transaction] = relationship(back_populates="predictions")
    model_version: Mapped[ModelVersion] = relationship()

    __table_args__ = (
        UniqueConstraint(
            "transaction_id", "model_version_id", name="uq_predictions_transaction_model"
        ),
        CheckConstraint(f"risk_level IN {RISK_LEVELS}", name="risk_level"),
    )


class User(Base):
    """An analyst or admin who logs in to the platform."""

    __tablename__ = "users"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    email: Mapped[str] = mapped_column(String(255), unique=True)  # stored lower-case
    full_name: Mapped[str] = mapped_column(String(255))
    password_hash: Mapped[str] = mapped_column(String(255))
    role: Mapped[str] = mapped_column(String(16))
    is_active: Mapped[bool] = mapped_column(Boolean, server_default="true")
    created_at: Mapped[dt.datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )
    last_login_at: Mapped[dt.datetime | None] = mapped_column(DateTime(timezone=True))

    __table_args__ = (CheckConstraint(f"role IN {USER_ROLES}", name="role"),)


class RefreshToken(Base):
    """One issued refresh token. Only its SHA-256 hash is stored; rotation links old -> new."""

    __tablename__ = "refresh_tokens"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True)
    user_id: Mapped[int] = mapped_column(ForeignKey("users.id", ondelete="CASCADE"), index=True)
    token_hash: Mapped[str] = mapped_column(String(64), unique=True)
    expires_at: Mapped[dt.datetime] = mapped_column(DateTime(timezone=True))
    created_at: Mapped[dt.datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )
    revoked_at: Mapped[dt.datetime | None] = mapped_column(DateTime(timezone=True))
    replaced_by_id: Mapped[int | None] = mapped_column(
        ForeignKey("refresh_tokens.id", ondelete="SET NULL")
    )

    user: Mapped[User] = relationship()


class AuditLog(Base):
    """Append-only record of every state-changing action (a database trigger refuses
    UPDATE, DELETE and TRUNCATE). actor_id is NULL for system actions (scripts)."""

    __tablename__ = "audit_logs"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True)
    # No ON DELETE action: a user with audit history cannot be deleted (deactivate instead).
    actor_id: Mapped[int | None] = mapped_column(ForeignKey("users.id"), index=True)
    action: Mapped[str] = mapped_column(String(64))
    entity_type: Mapped[str | None] = mapped_column(String(32))
    entity_id: Mapped[str | None] = mapped_column(String(64))
    # none_as_null: "no data" is SQL NULL, not a JSON null value.
    before: Mapped[dict[str, Any] | None] = mapped_column(JSONB(none_as_null=True))
    after: Mapped[dict[str, Any] | None] = mapped_column(JSONB(none_as_null=True))
    ip_address: Mapped[str | None] = mapped_column(String(64))
    created_at: Mapped[dt.datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )

    __table_args__ = (
        Index("ix_audit_logs_entity", "entity_type", "entity_id"),
        Index("ix_audit_logs_created_at", "created_at"),
        Index("ix_audit_logs_action", "action"),
    )


class Case(Base):
    """A transaction waiting for (or after) human review. One case per transaction.

    Queue order: priority_tier asc (1 = model alert, 2 = unusual pattern), then
    priority_score desc (expected loss = calibrated probability x amount).
    """

    __tablename__ = "cases"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True)
    transaction_id: Mapped[int] = mapped_column(
        ForeignKey("transactions.id", ondelete="CASCADE"), unique=True
    )
    prediction_id: Mapped[int] = mapped_column(ForeignKey("predictions.id"), index=True)
    reason: Mapped[str] = mapped_column(String(16))
    priority_tier: Mapped[int] = mapped_column(Integer)
    priority_score: Mapped[float] = mapped_column(Float)
    status: Mapped[str] = mapped_column(String(16), server_default="new")
    assigned_to: Mapped[int | None] = mapped_column(ForeignKey("users.id"), index=True)
    created_at: Mapped[dt.datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )
    updated_at: Mapped[dt.datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )
    closed_at: Mapped[dt.datetime | None] = mapped_column(DateTime(timezone=True))

    transaction: Mapped[Transaction] = relationship()
    prediction: Mapped[Prediction] = relationship()
    assignee: Mapped[User | None] = relationship()
    decisions: Mapped[list[Decision]] = relationship(back_populates="case", order_by="Decision.id")

    __table_args__ = (
        Index("ix_cases_queue", "status", "priority_tier", text("priority_score DESC")),
        CheckConstraint(f"reason IN {CASE_REASONS}", name="reason"),
        CheckConstraint(f"status IN {CASE_STATUSES}", name="status"),
        CheckConstraint("priority_tier IN (1, 2)", name="priority_tier"),
    )


class Decision(Base):
    """An analyst's decision on a case. Several per case (escalation chain); the latest
    confirmed_fraud / false_positive one is the final label."""

    __tablename__ = "decisions"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True)
    case_id: Mapped[int] = mapped_column(ForeignKey("cases.id", ondelete="CASCADE"), index=True)
    analyst_id: Mapped[int] = mapped_column(ForeignKey("users.id"), index=True)
    decision: Mapped[str] = mapped_column(String(16))
    comment: Mapped[str | None] = mapped_column(Text)
    created_at: Mapped[dt.datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )

    case: Mapped[Case] = relationship(back_populates="decisions")
    analyst: Mapped[User] = relationship()

    __table_args__ = (CheckConstraint(f"decision IN {DECISIONS}", name="decision"),)


class Setting(Base):
    """An admin-editable setting (keys: `thresholds`, `costs`). A missing key = defaults."""

    __tablename__ = "settings"

    key: Mapped[str] = mapped_column(String(64), primary_key=True)
    value: Mapped[dict[str, Any]] = mapped_column(JSONB)
    updated_by: Mapped[int | None] = mapped_column(ForeignKey("users.id"))
    updated_at: Mapped[dt.datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )


class MetricSnapshot(Base):
    """Performance of one model on one replayed day (labels revealed for that day).

    Counts are stored so that weeks and periods can be summed exactly; PR-AUC is per day.
    """

    __tablename__ = "metric_snapshots"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True)
    model_version_id: Mapped[int] = mapped_column(ForeignKey("model_versions.id"))
    day: Mapped[dt.date] = mapped_column(Date)
    label_source: Mapped[str] = mapped_column(String(16), server_default="ground_truth")
    n: Mapped[int] = mapped_column(Integer)
    n_fraud: Mapped[int] = mapped_column(Integer)
    alerts: Mapped[int] = mapped_column(Integer)
    tp: Mapped[int] = mapped_column(Integer)
    fp: Mapped[int] = mapped_column(Integer)
    fn: Mapped[int] = mapped_column(Integer)
    unusual_flags: Mapped[int] = mapped_column(Integer)
    precision: Mapped[float | None] = mapped_column(Float)
    recall: Mapped[float | None] = mapped_column(Float)
    pr_auc: Mapped[float | None] = mapped_column(Float)
    precision_at_capacity: Mapped[float | None] = mapped_column(Float)
    fraud_value: Mapped[float] = mapped_column(Float)  # USD of all frauds that day
    fraud_value_caught: Mapped[float] = mapped_column(Float)  # USD of alerted frauds
    total_cost: Mapped[float] = mapped_column(Float)
    no_model_cost: Mapped[float] = mapped_column(Float)
    net_savings: Mapped[float] = mapped_column(Float)
    created_at: Mapped[dt.datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )

    __table_args__ = (
        UniqueConstraint(
            "model_version_id", "day", "label_source", name="uq_metric_snapshots_model_day"
        ),
        # Only revealed ground truth for now; analyst decisions may become a second source.
        CheckConstraint("label_source IN ('ground_truth')", name="label_source"),
    )


class DriftReport(Base):
    """Drift of one feature (or `__score__`) on one day against the training reference."""

    __tablename__ = "drift_reports"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True)
    model_version_id: Mapped[int] = mapped_column(ForeignKey("model_versions.id"))
    day: Mapped[dt.date] = mapped_column(Date)
    feature_name: Mapped[str] = mapped_column(String(64))
    n: Mapped[int] = mapped_column(Integer)
    psi: Mapped[float] = mapped_column(Float)
    ks_stat: Mapped[float | None] = mapped_column(Float)
    ks_pvalue: Mapped[float | None] = mapped_column(Float)
    severity: Mapped[str] = mapped_column(String(16))
    created_at: Mapped[dt.datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )

    __table_args__ = (
        UniqueConstraint(
            "model_version_id", "day", "feature_name", name="uq_drift_reports_model_day_feature"
        ),
        Index("ix_drift_reports_day", "day"),
        CheckConstraint(f"severity IN {DRIFT_SEVERITIES}", name="severity"),
    )
