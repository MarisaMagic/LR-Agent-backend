"""agent session annotation_project_id

Revision ID: 006
Revises: 005
Create Date: 2026-06-03

"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy import inspect

revision: str = "006"
down_revision: str | None = "005"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def _column_exists(table_name: str, column_name: str) -> bool:
    bind = op.get_bind()
    columns = inspect(bind).get_columns(table_name)
    return any(column["name"] == column_name for column in columns)


def upgrade() -> None:
    if not _column_exists("agent_sessions", "annotation_project_id"):
        op.add_column(
            "agent_sessions",
            sa.Column("annotation_project_id", sa.String(length=64), nullable=True),
        )
        op.create_index(
            "ix_agent_sessions_user_project_updated",
            "agent_sessions",
            ["user_id", "annotation_project_id", "updated_at"],
        )


def downgrade() -> None:
    if _column_exists("agent_sessions", "annotation_project_id"):
        op.drop_index("ix_agent_sessions_user_project_updated", table_name="agent_sessions")
        op.drop_column("agent_sessions", "annotation_project_id")
