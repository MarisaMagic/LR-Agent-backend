"""Tests for workspace grep and list directory tools."""

from __future__ import annotations

from pathlib import Path

import pytest

from app.agent.tools.workspace_path import resolve_workspace_directory
from app.agent.tools.workspace_search import grep_workspace, list_workspace_directory
from app.core.config import Settings
from app.schemas.agent import ClientContextInput


@pytest.fixture
def workspace(tmp_path: Path) -> Path:
    root = tmp_path / "workspace"
    root.mkdir()
    (root / "src").mkdir()
    (root / "src" / "main.py").write_text(
        "def hello():\n    print('hello')\n\nclass Foo:\n    pass\n",
        encoding="utf-8",
    )
    (root / "src" / "utils.ts").write_text(
        "export const AgentMode = 'ask';\n",
        encoding="utf-8",
    )
    (root / "notes.md").write_text("# Title\n", encoding="utf-8")
    (root / ".git").mkdir()
    (root / ".git" / "config").write_text("[core]\n", encoding="utf-8")
    (root / "secret").mkdir()
    (root / "secret" / "outside.txt").write_text("hidden", encoding="utf-8")
    return root


@pytest.fixture
def client_context(workspace: Path) -> ClientContextInput:
    return ClientContextInput(
        workspace_root=str(workspace),
        active_file_path=str(workspace / "src" / "main.py"),
        active_relative_path="src/main.py",
    )


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


def test_grep_finds_matches(client_context: ClientContextInput, settings: Settings) -> None:
    result = grep_workspace(client_context, "class Foo", settings=settings)
    assert "src/main.py" in result
    assert "class Foo" in result


def test_grep_no_match(client_context: ClientContextInput, settings: Settings) -> None:
    result = grep_workspace(client_context, "NonExistentSymbolXYZ", settings=settings)
    assert "未找到匹配" in result


def test_grep_glob_filter(client_context: ClientContextInput, settings: Settings) -> None:
    result = grep_workspace(
        client_context,
        "AgentMode",
        glob_pattern="*.ts",
        settings=settings,
    )
    assert "utils.ts" in result
    assert "main.py" not in result


def test_grep_scoped_to_directory(client_context: ClientContextInput, settings: Settings) -> None:
    result = grep_workspace(client_context, "hello", path="src", settings=settings)
    assert "src/main.py" in result
    assert "notes.md" not in result


def test_grep_max_results(client_context: ClientContextInput, settings: Settings) -> None:
    result = grep_workspace(
        client_context,
        ".",
        max_results=1,
        settings=settings,
    )
    assert "结果已截断" in result


def test_grep_rejects_traversal(client_context: ClientContextInput, settings: Settings) -> None:
    result = grep_workspace(
        client_context,
        "hidden",
        path="../secret",
        settings=settings,
    )
    assert ".." in result or "未找到" in result or "路径" in result


def test_grep_skips_git_dir(client_context: ClientContextInput, settings: Settings) -> None:
    result = grep_workspace(client_context, "\\[core\\]", settings=settings)
    assert ".git/config" not in result


def test_list_directory_root(client_context: ClientContextInput, settings: Settings) -> None:
    result = list_workspace_directory(client_context, "", settings=settings)
    assert "src" in result
    assert "directory" in result
    assert "notes.md" in result
    assert ".git" not in result


def test_list_directory_subdir(client_context: ClientContextInput, settings: Settings) -> None:
    result = list_workspace_directory(client_context, "src", settings=settings)
    assert "main.py" in result
    assert "utils.ts" in result


def test_list_directory_rejects_traversal(client_context: ClientContextInput, settings: Settings) -> None:
    result = list_workspace_directory(client_context, "../secret", settings=settings)
    assert ".." in result or "未找到" in result


def test_resolve_workspace_directory_empty_path(
    client_context: ClientContextInput,
    workspace: Path,
) -> None:
    resolved, err = resolve_workspace_directory(client_context, "")
    assert err == ""
    assert resolved is not None
    assert resolved.resolve() == workspace.resolve()
