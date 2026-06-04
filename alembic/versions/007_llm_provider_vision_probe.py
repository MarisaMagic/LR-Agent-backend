"""Add vision probe fields to llm_providers.

Revision ID: 007
Revises: 006
"""
from __future__ import annotations

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

revision: str = "007"
down_revision: Union[str, None] = "006"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column(
        "llm_providers",
        sa.Column("supports_vision", sa.Boolean(), nullable=False, server_default=sa.false()),
    )
    op.add_column(
        "llm_providers",
        sa.Column("vision_probed_at", sa.DateTime(timezone=True), nullable=True),
    )
    op.add_column(
        "llm_providers",
        sa.Column("vision_probe_detail", sa.String(length=512), nullable=False, server_default=""),
    )


def downgrade() -> None:
    op.drop_column("llm_providers", "vision_probe_detail")
    op.drop_column("llm_providers", "vision_probed_at")
    op.drop_column("llm_providers", "supports_vision")
