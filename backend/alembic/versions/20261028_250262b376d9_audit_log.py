"""audit log

Revision ID: 250262b376d9
Revises: a8b33a4a1f71
Create Date: 2026-10-28 10:44:56.725974
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

# Append-only: the database itself refuses to change or remove audit rows.
APPEND_ONLY_FUNCTION = """
CREATE FUNCTION audit_logs_append_only() RETURNS trigger LANGUAGE plpgsql AS $$
BEGIN
    RAISE EXCEPTION 'audit_logs is append-only (% refused)', TG_OP;
END;
$$
"""

revision: str = "250262b376d9"
down_revision: str | None = "a8b33a4a1f71"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "audit_logs",
        sa.Column("id", sa.BigInteger(), nullable=False),
        sa.Column("actor_id", sa.Integer(), nullable=True),
        sa.Column("action", sa.String(length=64), nullable=False),
        sa.Column("entity_type", sa.String(length=32), nullable=True),
        sa.Column("entity_id", sa.String(length=64), nullable=True),
        sa.Column("before", postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column("after", postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column("ip_address", sa.String(length=64), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.ForeignKeyConstraint(
            ["actor_id"], ["users.id"], name=op.f("fk_audit_logs_actor_id_users")
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_audit_logs")),
    )
    op.create_index("ix_audit_logs_action", "audit_logs", ["action"], unique=False)
    op.create_index(op.f("ix_audit_logs_actor_id"), "audit_logs", ["actor_id"], unique=False)
    op.create_index("ix_audit_logs_created_at", "audit_logs", ["created_at"], unique=False)
    op.create_index(
        "ix_audit_logs_entity", "audit_logs", ["entity_type", "entity_id"], unique=False
    )

    op.execute(APPEND_ONLY_FUNCTION)
    op.execute(
        "CREATE TRIGGER audit_logs_no_update_delete BEFORE UPDATE OR DELETE ON audit_logs "
        "FOR EACH ROW EXECUTE FUNCTION audit_logs_append_only()"
    )
    op.execute(
        "CREATE TRIGGER audit_logs_no_truncate BEFORE TRUNCATE ON audit_logs "
        "FOR EACH STATEMENT EXECUTE FUNCTION audit_logs_append_only()"
    )


def downgrade() -> None:
    op.execute("DROP TRIGGER IF EXISTS audit_logs_no_truncate ON audit_logs")
    op.execute("DROP TRIGGER IF EXISTS audit_logs_no_update_delete ON audit_logs")
    op.execute("DROP FUNCTION IF EXISTS audit_logs_append_only()")
    op.drop_index("ix_audit_logs_entity", table_name="audit_logs")
    op.drop_index("ix_audit_logs_created_at", table_name="audit_logs")
    op.drop_index(op.f("ix_audit_logs_actor_id"), table_name="audit_logs")
    op.drop_index("ix_audit_logs_action", table_name="audit_logs")
    op.drop_table("audit_logs")
