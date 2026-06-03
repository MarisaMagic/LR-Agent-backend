"""agent chat sort_index and soft delete

Revision ID: 004
Revises: 003
Create Date: 2026-06-03

"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy import inspect


revision: str = "004"
down_revision: str | None = "003"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def _table_exists(table_name: str) -> bool:
    bind = op.get_bind()
    return table_name in inspect(bind).get_table_names()


def _column_exists(table_name: str, column_name: str) -> bool:
    bind = op.get_bind()
    columns = inspect(bind).get_columns(table_name)
    return any(column["name"] == column_name for column in columns)


def _index_exists(table_name: str, index_name: str) -> bool:
    bind = op.get_bind()
    indexes = inspect(bind).get_indexes(table_name)
    return any(index["name"] == index_name for index in indexes)


def upgrade() -> None:
    if not _table_exists("agent_messages") or not _table_exists("agent_sessions"):
        return

    if not _column_exists("agent_messages", "sort_index"):
        op.add_column(
            "agent_messages",
            sa.Column("sort_index", sa.Integer(), nullable=False, server_default="0"),
        )

    if not _column_exists("agent_sessions", "deleted_at"):
        op.add_column(
            "agent_sessions",
            sa.Column("deleted_at", sa.DateTime(timezone=True), nullable=True),
        )

    if _column_exists("agent_messages", "sort_index"):
        op.execute(
            """
            WITH ranked AS (
                SELECT id,
                       ROW_NUMBER() OVER (
                           PARTITION BY session_id
                           ORDER BY created_at ASC, id ASC
                       ) - 1 AS rn
                FROM agent_messages
            )
            UPDATE agent_messages m
            SET sort_index = ranked.rn
            FROM ranked
            WHERE m.id = ranked.id
            """
        )

    if not _index_exists("agent_messages", "idx_agent_messages_session_sort"):
        op.create_index(
            "idx_agent_messages_session_sort",
            "agent_messages",
            ["session_id", "sort_index"],
            unique=False,
            if_not_exists=True,
        )


def downgrade() -> None:
    if _table_exists("agent_messages") and _index_exists(
        "agent_messages",
        "idx_agent_messages_session_sort",
    ):
        op.drop_index(
            "idx_agent_messages_session_sort",
            table_name="agent_messages",
            if_exists=True,
        )

    if _table_exists("agent_sessions") and _column_exists("agent_sessions", "deleted_at"):
        op.drop_column("agent_sessions", "deleted_at")

    if _table_exists("agent_messages") and _column_exists("agent_messages", "sort_index"):
        op.drop_column("agent_messages", "sort_index")
