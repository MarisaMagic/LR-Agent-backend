"""Drop legacy agent session and llm provider tables.

Revision ID: 010
Revises: 009
Create Date: 2026-07-26

"""

from collections.abc import Sequence

from alembic import op
from sqlalchemy import inspect

revision: str = "010"
down_revision: str | None = "009"
branch_labels: Sequence[str] | None = None
depends_on: Sequence[str] | None = None


def _index_exists(table_name: str, index_name: str) -> bool:
    bind = op.get_bind()
    inspector = inspect(bind)
    return any(index["name"] == index_name for index in inspector.get_indexes(table_name))


def _table_exists(table_name: str) -> bool:
    bind = op.get_bind()
    inspector = inspect(bind)
    return table_name in inspector.get_table_names()


def upgrade() -> None:
    if _table_exists("agent_messages"):
        if _index_exists("agent_messages", "idx_agent_messages_session_sort"):
            op.drop_index("idx_agent_messages_session_sort", table_name="agent_messages")
        if _index_exists("agent_messages", "idx_agent_messages_session_id"):
            op.drop_index("idx_agent_messages_session_id", table_name="agent_messages")
        op.drop_table("agent_messages")

    if _table_exists("agent_sessions"):
        if _index_exists("agent_sessions", "ix_agent_sessions_user_project_updated"):
            op.drop_index("ix_agent_sessions_user_project_updated", table_name="agent_sessions")
        if _index_exists("agent_sessions", "idx_agent_sessions_user_id"):
            op.drop_index("idx_agent_sessions_user_id", table_name="agent_sessions")
        op.drop_table("agent_sessions")

    if _table_exists("llm_providers"):
        if _index_exists("llm_providers", "idx_llm_providers_user_id"):
            op.drop_index("idx_llm_providers_user_id", table_name="llm_providers")
        op.drop_table("llm_providers")


def downgrade() -> None:
    # Legacy tables removed intentionally; downgrade is a no-op.
    pass
