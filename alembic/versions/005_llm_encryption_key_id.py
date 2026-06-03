"""llm provider encryption_key_id

Revision ID: 005
Revises: 004
Create Date: 2026-06-03

"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy import inspect

revision: str = "005"
down_revision: str | None = "004"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def _column_exists(table_name: str, column_name: str) -> bool:
    bind = op.get_bind()
    columns = inspect(bind).get_columns(table_name)
    return any(column["name"] == column_name for column in columns)


def upgrade() -> None:
    if not _column_exists("llm_providers", "encryption_key_id"):
        op.add_column(
            "llm_providers",
            sa.Column(
                "encryption_key_id",
                sa.String(length=8),
                nullable=False,
                server_default="v0",
            ),
        )


def downgrade() -> None:
    if _column_exists("llm_providers", "encryption_key_id"):
        op.drop_column("llm_providers", "encryption_key_id")
