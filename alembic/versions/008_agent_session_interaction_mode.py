"""agent session interaction_mode

Revision ID: 008
Revises: 007
Create Date: 2026-06-03

"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy import inspect

revision: str = "008"
down_revision: str | None = "007"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def _column_exists(table_name: str, column_name: str) -> bool:
    bind = op.get_bind()
    columns = inspect(bind).get_columns(table_name)
    return any(column["name"] == column_name for column in columns)


def upgrade() -> None:
    if not _column_exists("agent_sessions", "interaction_mode"):
        op.add_column(
            "agent_sessions",
            sa.Column("interaction_mode", sa.String(length=16), nullable=True),
        )


def downgrade() -> None:
    if _column_exists("agent_sessions", "interaction_mode"):
        op.drop_column("agent_sessions", "interaction_mode")
