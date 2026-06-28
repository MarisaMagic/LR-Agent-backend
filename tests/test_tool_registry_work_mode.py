"""Tests for editor/annotation work mode tool filtering."""

from __future__ import annotations

import uuid

import pytest

from app.agent.tools.registry import ANNOTATION_TOOL_NAMES, build_tools_p1
from app.core.config import Settings
from app.models.user import User
from app.schemas.agent import ClientContextInput


@pytest.fixture
def settings(monkeypatch: pytest.MonkeyPatch) -> Settings:
    monkeypatch.setenv("SECRET_KEY", "test-secret-key-for-unit-tests")
    monkeypatch.setenv("POSTGRES_PASSWORD", "test")
    monkeypatch.setenv(
        "DATABASE_URL",
        "postgresql+asyncpg://lr_agent:test@localhost:5432/lr_agent",
    )
    monkeypatch.setenv("REDIS_PASSWORD", "test")
    monkeypatch.setenv("REDIS_URL", "redis://:test@localhost:6379/0")
    return Settings()


@pytest.fixture
def user() -> User:
    return User(
        id=uuid.uuid4(),
        email="test@example.com",
        email_verified=True,
        username="tester",
        display_name="Tester",
        password_hash="x",
        is_active=True,
    )


def test_editor_mode_excludes_annotation_tools(user: User, settings: Settings) -> None:
    ctx = ClientContextInput(
        workspace_root="/tmp/ws",
        work_mode="editor",
    )
    names = {tool.name for tool in build_tools_p1(user, ctx, settings=settings)}
    assert ANNOTATION_TOOL_NAMES.isdisjoint(names)


def test_annotation_mode_includes_annotation_tools(
    user: User, settings: Settings
) -> None:
    ctx = ClientContextInput(
        workspace_root="/tmp/ws",
        work_mode="annotation",
    )
    names = {tool.name for tool in build_tools_p1(user, ctx, settings=settings)}
    assert ANNOTATION_TOOL_NAMES.issubset(names)
