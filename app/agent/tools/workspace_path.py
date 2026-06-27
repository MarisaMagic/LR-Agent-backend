"""工作区 / 项目文件路径解析与安全校验。

所有文件读取工具（workspace_file_reader、assist_vision）的统一入口：
  1. 确定允许访问的根目录（workspace_root + project_directory）
  2. 解析用户/LLM 传入的路径，回退到当前打开文件
  3. 校验路径在根目录内、禁止 .. 穿越
"""

from __future__ import annotations

from pathlib import Path

from app.agent.context_helpers import project_directory
from app.schemas.agent import ClientContextInput


def normalize_relative_path(relative_path: str) -> str:
    """将相对路径统一为正斜杠格式，去除空段与 `.`。"""
    return "/".join(part for part in relative_path.replace("\\", "/").split("/") if part and part != ".")


def allowed_roots(client_context: ClientContextInput | None) -> list[Path]:
    """返回可访问的根目录列表（工作区根 + 项目目录，去重）。"""
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
    """判断 candidate 是否在 root 目录树下。"""
    try:
        candidate.relative_to(root)
        return True
    except ValueError:
        return False


def resolve_workspace_file(
    client_context: ClientContextInput | None,
    path: str,
) -> tuple[Path | None, str]:
    """解析并校验文件路径，返回 (绝对路径, 错误信息)。

    - path 为空时回退到 active_file_path / active_relative_path
    - 支持绝对路径（须在 allowed_roots 内）与相对路径
    - 禁止 .. 目录穿越
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


def resolve_workspace_write_path(
    client_context: ClientContextInput | None,
    path: str,
) -> tuple[Path | None, str]:
    """解析并校验写文件目标路径，返回 (绝对路径, 错误信息)。

    与 resolve_workspace_file 的差异：目标文件不要求已存在，仅校验：
    - 路径在 allowed_roots 内
    - 无 .. 目录穿越
    - 父目录不存在时由前端/Electron ensureDir 自动创建
    """
    raw = (path or "").strip()
    if not raw:
        return None, "请提供写入目标的相对路径（如 reports/summary.md）。"

    roots = allowed_roots(client_context)
    if not roots:
        return None, "未绑定工作区或项目目录，无法写入本地文件。"

    candidate_input = Path(raw)
    if candidate_input.is_absolute():
        try:
            candidate = candidate_input.resolve()
        except OSError as exc:
            return None, f"路径无效：{exc}"
        for root in roots:
            if _is_under_root(candidate, root):
                return candidate, ""
        return None, "目标路径不在当前工作区或项目目录内。"

    rel = normalize_relative_path(raw)
    if ".." in rel.split("/"):
        return None, "路径不能包含 .."

    for root in roots:
        candidate = (root / rel).resolve()
        if not _is_under_root(candidate, root):
            continue
        return candidate, ""
    return None, f"无法解析写入路径：{rel}"
