"""Resolve and validate workspace / project file paths for agent read tools."""

from __future__ import annotations

from pathlib import Path

from app.agent.context_helpers import project_directory
from app.schemas.agent import ClientContextInput


def normalize_relative_path(relative_path: str) -> str:
    return "/".join(part for part in relative_path.replace("\\", "/").split("/") if part and part != ".")


def allowed_roots(client_context: ClientContextInput | None) -> list[Path]:
    roots: list[Path] = []
    seen: set[str] = set()
    if client_context is None:
        return roots

    for raw in (
        (client_context.workspace_root or "").strip(),
        (project_directory(client_context) or "").strip(),
    ):
        if not raw:
            continue
        try:
            resolved = str(Path(raw).resolve())
        except OSError:
            continue
        if resolved in seen:
            continue
        seen.add(resolved)
        roots.append(Path(resolved))
    return roots


def _is_under_root(candidate: Path, root: Path) -> bool:
    try:
        candidate.relative_to(root)
        return True
    except ValueError:
        return False


def resolve_workspace_file(
    client_context: ClientContextInput | None,
    path: str,
) -> tuple[Path | None, str]:
    """
    Resolve a user-supplied path to an absolute file under workspace_root or project_directory.
    Empty path falls back to active_file_path / active_relative_path.
    """
    raw = (path or "").strip()
    if not raw and client_context is not None:
        active_abs = (client_context.active_file_path or "").strip()
        if active_abs:
            raw = active_abs
        else:
            rel = normalize_relative_path(client_context.active_relative_path or "")
            if rel:
                raw = rel

    if not raw:
        return None, "请提供相对路径，或在工作区中打开目标文件后再提问。"

    roots = allowed_roots(client_context)
    if not roots:
        return None, "未绑定工作区或项目目录，无法读取本地文件。"

    candidate_input = Path(raw)
    if candidate_input.is_absolute():
        try:
            candidate = candidate_input.resolve()
        except OSError as exc:
            return None, f"路径无效：{exc}"
        for root in roots:
            if _is_under_root(candidate, root) and candidate.is_file():
                return candidate, ""
        return None, "文件不在当前工作区或项目目录内。"

    rel = normalize_relative_path(raw)
    if ".." in rel.split("/"):
        return None, "路径不能包含 .."

    for root in roots:
        candidate = (root / rel).resolve()
        if not _is_under_root(candidate, root):
            continue
        if candidate.is_file():
            return candidate, ""
    return None, f"未找到文件：{rel}"
