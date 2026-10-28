"""review queue and settings

Revision ID: 1983fca8e61f
Revises: 250262b376d9
Create Date: 2026-10-28 13:27:27.486424
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = "1983fca8e61f"
down_revision: str | None = "250262b376d9"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "settings",
        sa.Column("key", sa.String(length=64), nullable=False),
        sa.Column("value", postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.Column("updated_by", sa.Integer(), nullable=True),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.ForeignKeyConstraint(
            ["updated_by"], ["users.id"], name=op.f("fk_settings_updated_by_users")
        ),
        sa.PrimaryKeyConstraint("key", name=op.f("pk_settings")),
    )
    op.create_table(
        "cases",
        sa.Column("id", sa.BigInteger(), nullable=False),
        sa.Column("transaction_id", sa.BigInteger(), nullable=False),
        sa.Column("prediction_id", sa.BigInteger(), nullable=False),
        sa.Column("reason", sa.String(length=16), nullable=False),
        sa.Column("priority_tier", sa.Integer(), nullable=False),
        sa.Column("priority_score", sa.Float(), nullable=False),
        sa.Column("status", sa.String(length=16), server_default="new", nullable=False),
        sa.Column("assigned_to", sa.Integer(), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.Column("closed_at", sa.DateTime(timezone=True), nullable=True),
        sa.CheckConstraint("reason IN ('model_score', 'anomaly')", name=op.f("ck_cases_reason")),
        sa.CheckConstraint(
            "status IN ('new', 'assigned', 'escalated', 'closed')", name=op.f("ck_cases_status")
        ),
        sa.CheckConstraint("priority_tier IN (1, 2)", name=op.f("ck_cases_priority_tier")),
        sa.ForeignKeyConstraint(
            ["assigned_to"], ["users.id"], name=op.f("fk_cases_assigned_to_users")
        ),
        sa.ForeignKeyConstraint(
            ["prediction_id"], ["predictions.id"], name=op.f("fk_cases_prediction_id_predictions")
        ),
        sa.ForeignKeyConstraint(
            ["transaction_id"],
            ["transactions.id"],
            name=op.f("fk_cases_transaction_id_transactions"),
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_cases")),
        sa.UniqueConstraint("transaction_id", name=op.f("uq_cases_transaction_id")),
    )
    op.create_index(op.f("ix_cases_assigned_to"), "cases", ["assigned_to"], unique=False)
    op.create_index(op.f("ix_cases_prediction_id"), "cases", ["prediction_id"], unique=False)
    op.create_index(
        "ix_cases_queue",
        "cases",
        ["status", "priority_tier", sa.literal_column("priority_score DESC")],
        unique=False,
    )
    op.create_table(
        "decisions",
        sa.Column("id", sa.BigInteger(), nullable=False),
        sa.Column("case_id", sa.BigInteger(), nullable=False),
        sa.Column("analyst_id", sa.Integer(), nullable=False),
        sa.Column("decision", sa.String(length=16), nullable=False),
        sa.Column("comment", sa.Text(), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.CheckConstraint(
            "decision IN ('confirmed_fraud', 'false_positive', 'escalate')",
            name=op.f("ck_decisions_decision"),
        ),
        sa.ForeignKeyConstraint(
            ["analyst_id"], ["users.id"], name=op.f("fk_decisions_analyst_id_users")
        ),
        sa.ForeignKeyConstraint(
            ["case_id"], ["cases.id"], name=op.f("fk_decisions_case_id_cases"), ondelete="CASCADE"
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_decisions")),
    )
    op.create_index(op.f("ix_decisions_analyst_id"), "decisions", ["analyst_id"], unique=False)
    op.create_index(op.f("ix_decisions_case_id"), "decisions", ["case_id"], unique=False)


def downgrade() -> None:
    op.drop_index(op.f("ix_decisions_case_id"), table_name="decisions")
    op.drop_index(op.f("ix_decisions_analyst_id"), table_name="decisions")
    op.drop_table("decisions")
    op.drop_index("ix_cases_queue", table_name="cases")
    op.drop_index(op.f("ix_cases_prediction_id"), table_name="cases")
    op.drop_index(op.f("ix_cases_assigned_to"), table_name="cases")
    op.drop_table("cases")
    op.drop_table("settings")
