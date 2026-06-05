"""Tests for workspace path resolution and file read tools."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from app.agent.tools.workspace_file_reader import (
    VISION_PATH_MARKER,
    extract_vision_path_from_tool_result,
    read_image_for_vision_tool,
    read_workspace_text_file,
)
from app.agent.tools.workspace_path import allowed_roots, resolve_workspace_file
from app.core.config import Settings
from app.schemas.agent import ClientContextInput


@pytest.fixture
def workspace(tmp_path: Path) -> Path:
    root = tmp_path / "workspace"
    root.mkdir()
    (root / "src").mkdir()
    (root / "src" / "main.py").write_text("print('hello')\n", encoding="utf-8")
    (root / "notes.md").write_text("# Title\n", encoding="utf-8")
    (root / "secret").mkdir()
    (root / "secret" / "outside.txt").write_text("nope", encoding="utf-8")
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


def test_allowed_roots_deduplicates(workspace: Path) -> None:
    ctx = ClientContextInput(
        workspace_root=str(workspace),
        project_directory_path=str(workspace),
    )
    roots = allowed_roots(ctx)
    assert len(roots) == 1


def test_resolve_relative_path(client_context: ClientContextInput) -> None:
    resolved, err = resolve_workspace_file(client_context, "src/main.py")
    assert err == ""
    assert resolved is not None
    assert resolved.name == "main.py"


def test_resolve_active_file_when_path_empty(client_context: ClientContextInput) -> None:
    resolved, err = resolve_workspace_file(client_context, "")
    assert err == ""
    assert resolved is not None
    assert resolved.name == "main.py"


def test_reject_path_traversal(client_context: ClientContextInput) -> None:
    resolved, err = resolve_workspace_file(client_context, "../secret/outside.txt")
    assert resolved is None
    assert ".." in err or "未找到" in err


def test_read_workspace_text(client_context: ClientContextInput, settings: Settings) -> None:
    content = read_workspace_text_file(client_context, "notes.md", settings=settings)
    assert "Title" in content
    assert content.startswith("文件：")


def test_read_image_for_vision_returns_marker(
    workspace: Path,
) -> None:
    from PIL import Image

    img_path = workspace / "photo.jpg"
    Image.new("RGB", (8, 8), color=(255, 0, 0)).save(img_path, format="JPEG")
    ctx = ClientContextInput(workspace_root=str(workspace))
    raw = read_image_for_vision_tool(ctx, "photo.jpg", provider_is_vision=True)
    data = json.loads(raw)
    assert data["ok"] is True
    assert VISION_PATH_MARKER in data
    path = extract_vision_path_from_tool_result("read_image_for_vision", raw)
    assert path == str(img_path.resolve())


def test_read_image_requires_vision_model(client_context: ClientContextInput) -> None:
    result = read_image_for_vision_tool(
        client_context,
        "src/main.py",
        provider_is_vision=False,
    )
    assert "视觉" in result


def test_read_document_docx(
    workspace: Path,
    settings: Settings,
) -> None:
    pytest.importorskip("docx")
    from docx import Document

    doc_path = workspace / "readme.docx"
    doc = Document()
    doc.add_paragraph("Hello DOCX")
    doc.save(doc_path)

    from app.agent.tools.workspace_file_reader import read_document_file

    ctx = ClientContextInput(workspace_root=str(workspace))
    text = read_document_file(ctx, "readme.docx", settings=settings)
    assert "Hello DOCX" in text
