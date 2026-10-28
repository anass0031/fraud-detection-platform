"""initial scoring tables

Revision ID: 63fd379c69f4
Revises:
Create Date: 2026-10-28 09:49:17.302924
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = "63fd379c69f4"
down_revision: str | None = None
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "model_versions",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("version", sa.String(length=64), nullable=False),
        sa.Column("algorithm", sa.String(length=32), nullable=False),
        sa.Column("artifact_path", sa.String(length=255), nullable=False),
        sa.Column("anomaly_version", sa.String(length=64), nullable=True),
        sa.Column("feature_list", postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.Column("hyperparams", postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column("metrics", postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column("calibration_method", sa.String(length=32), nullable=True),
        sa.Column("policy", sa.String(length=32), nullable=False),
        sa.Column("threshold", sa.Float(), nullable=False),
        sa.Column("status", sa.String(length=16), server_default="candidate", nullable=False),
        sa.Column("trained_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("training_data_window", postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column("notes", sa.Text(), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_model_versions")),
        sa.UniqueConstraint("version", name=op.f("uq_model_versions_version")),
        sa.CheckConstraint(
            "status IN ('candidate', 'active', 'archived')", name=op.f("ck_model_versions_status")
        ),
    )
    op.create_index(
        "uq_model_versions_one_active",
        "model_versions",
        ["status"],
        unique=True,
        postgresql_where=sa.text("status = 'active'"),
    )
    op.create_table(
        "transactions",
        sa.Column("id", sa.BigInteger(), nullable=False),
        sa.Column("external_id", sa.BigInteger(), nullable=False),
        sa.Column("transaction_dt", sa.BigInteger(), nullable=False),
        sa.Column("transaction_time", sa.DateTime(), nullable=False),
        sa.Column("amount", sa.Numeric(precision=12, scale=3), nullable=False),
        sa.Column("product_code", sa.String(length=8), nullable=True),
        sa.Column("card_network", sa.String(length=32), nullable=True),
        sa.Column("card_type", sa.String(length=32), nullable=True),
        sa.Column("payer_email_domain", sa.String(length=64), nullable=True),
        sa.Column("device_type", sa.String(length=16), nullable=True),
        sa.Column("customer_key", sa.String(length=128), nullable=True),
        sa.Column("raw", postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.Column("features", postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column("ground_truth", sa.Boolean(), nullable=True),
        sa.Column("ground_truth_revealed_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("source", sa.String(length=16), nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_transactions")),
        sa.UniqueConstraint("external_id", name=op.f("uq_transactions_external_id")),
        sa.CheckConstraint(
            "source IN ('history', 'batch', 'realtime', 'replay')",
            name=op.f("ck_transactions_source"),
        ),
    )
    op.create_index(
        "ix_transactions_customer_key_dt",
        "transactions",
        ["customer_key", "transaction_dt"],
        unique=False,
    )
    op.create_index("ix_transactions_source", "transactions", ["source"], unique=False)
    op.create_index(
        "ix_transactions_transaction_time", "transactions", ["transaction_time"], unique=False
    )
    op.create_table(
        "predictions",
        sa.Column("id", sa.BigInteger(), nullable=False),
        sa.Column("transaction_id", sa.BigInteger(), nullable=False),
        sa.Column("model_version_id", sa.Integer(), nullable=False),
        sa.Column("fraud_score", sa.Float(), nullable=False),
        sa.Column("expected_loss", sa.Float(), nullable=False),
        sa.Column("is_alert", sa.Boolean(), nullable=False),
        sa.Column("anomaly_score", sa.Float(), nullable=True),
        sa.Column("is_unusual", sa.Boolean(), server_default="false", nullable=False),
        sa.Column("risk_level", sa.String(length=16), nullable=False),
        sa.Column("shap_values", postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.Column("base_value", sa.Float(), nullable=True),
        sa.Column("latency_ms", sa.Float(), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.ForeignKeyConstraint(
            ["model_version_id"],
            ["model_versions.id"],
            name=op.f("fk_predictions_model_version_id_model_versions"),
        ),
        sa.ForeignKeyConstraint(
            ["transaction_id"],
            ["transactions.id"],
            name=op.f("fk_predictions_transaction_id_transactions"),
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_predictions")),
        sa.UniqueConstraint(
            "transaction_id", "model_version_id", name="uq_predictions_transaction_model"
        ),
        sa.CheckConstraint(
            "risk_level IN ('low', 'medium', 'high', 'critical')",
            name=op.f("ck_predictions_risk_level"),
        ),
    )
    op.create_index(
        op.f("ix_predictions_model_version_id"), "predictions", ["model_version_id"], unique=False
    )
    op.create_index(
        op.f("ix_predictions_transaction_id"), "predictions", ["transaction_id"], unique=False
    )


def downgrade() -> None:
    op.drop_index(op.f("ix_predictions_transaction_id"), table_name="predictions")
    op.drop_index(op.f("ix_predictions_model_version_id"), table_name="predictions")
    op.drop_table("predictions")
    op.drop_index("ix_transactions_transaction_time", table_name="transactions")
    op.drop_index("ix_transactions_source", table_name="transactions")
    op.drop_index("ix_transactions_customer_key_dt", table_name="transactions")
    op.drop_table("transactions")
    op.drop_index(
        "uq_model_versions_one_active",
        table_name="model_versions",
        postgresql_where=sa.text("status = 'active'"),
    )
    op.drop_table("model_versions")
