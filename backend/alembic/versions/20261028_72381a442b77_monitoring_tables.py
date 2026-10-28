"""monitoring tables

Revision ID: 72381a442b77
Revises: 1983fca8e61f
Create Date: 2026-10-28 15:04:23.396419
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "72381a442b77"
down_revision: str | None = "1983fca8e61f"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "drift_reports",
        sa.Column("id", sa.BigInteger(), nullable=False),
        sa.Column("model_version_id", sa.Integer(), nullable=False),
        sa.Column("day", sa.Date(), nullable=False),
        sa.Column("feature_name", sa.String(length=64), nullable=False),
        sa.Column("n", sa.Integer(), nullable=False),
        sa.Column("psi", sa.Float(), nullable=False),
        sa.Column("ks_stat", sa.Float(), nullable=True),
        sa.Column("ks_pvalue", sa.Float(), nullable=True),
        sa.Column("severity", sa.String(length=16), nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.CheckConstraint(
            "severity IN ('stable', 'moderate', 'significant')",
            name=op.f("ck_drift_reports_severity"),
        ),
        sa.ForeignKeyConstraint(
            ["model_version_id"],
            ["model_versions.id"],
            name=op.f("fk_drift_reports_model_version_id_model_versions"),
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_drift_reports")),
        sa.UniqueConstraint(
            "model_version_id", "day", "feature_name", name="uq_drift_reports_model_day_feature"
        ),
    )
    op.create_index("ix_drift_reports_day", "drift_reports", ["day"], unique=False)
    op.create_table(
        "metric_snapshots",
        sa.Column("id", sa.BigInteger(), nullable=False),
        sa.Column("model_version_id", sa.Integer(), nullable=False),
        sa.Column("day", sa.Date(), nullable=False),
        sa.Column(
            "label_source", sa.String(length=16), server_default="ground_truth", nullable=False
        ),
        sa.Column("n", sa.Integer(), nullable=False),
        sa.Column("n_fraud", sa.Integer(), nullable=False),
        sa.Column("alerts", sa.Integer(), nullable=False),
        sa.Column("tp", sa.Integer(), nullable=False),
        sa.Column("fp", sa.Integer(), nullable=False),
        sa.Column("fn", sa.Integer(), nullable=False),
        sa.Column("unusual_flags", sa.Integer(), nullable=False),
        sa.Column("precision", sa.Float(), nullable=True),
        sa.Column("recall", sa.Float(), nullable=True),
        sa.Column("pr_auc", sa.Float(), nullable=True),
        sa.Column("precision_at_capacity", sa.Float(), nullable=True),
        sa.Column("fraud_value", sa.Float(), nullable=False),
        sa.Column("fraud_value_caught", sa.Float(), nullable=False),
        sa.Column("total_cost", sa.Float(), nullable=False),
        sa.Column("no_model_cost", sa.Float(), nullable=False),
        sa.Column("net_savings", sa.Float(), nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.CheckConstraint(
            "label_source IN ('ground_truth')", name=op.f("ck_metric_snapshots_label_source")
        ),
        sa.ForeignKeyConstraint(
            ["model_version_id"],
            ["model_versions.id"],
            name=op.f("fk_metric_snapshots_model_version_id_model_versions"),
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_metric_snapshots")),
        sa.UniqueConstraint(
            "model_version_id", "day", "label_source", name="uq_metric_snapshots_model_day"
        ),
    )


def downgrade() -> None:
    op.drop_table("metric_snapshots")
    op.drop_index("ix_drift_reports_day", table_name="drift_reports")
    op.drop_table("drift_reports")
