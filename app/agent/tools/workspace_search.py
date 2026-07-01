"""工作区代码/文本搜索与目录列表（grep、list_dir）。"""

from __future__ import annotations

import fnmatch
import re
from pathlib import Path

from app.agent.tools.workspace_file_reader import TEXT_BLOCKLIST_SUFFIXES
from app.agent.tools.workspace_path import (
    allowed_roots,
    normalize_relative_path,
    relative_path_from_roots,
    resolve_workspace_directory,
    resolve_workspace_file,
)
from app.core.config import Settings
from app.schemas.agent import ClientContextInput

SKIP_DIR_NAMES = frozenset(
    {
        ".git",
        ".lr-agent",
        "node_modules",
        "__pycache__",
        ".venv",
        "venv",
        ".mypy_cache",
        ".pytest_cache",
        "dist",
        "build",
    }
)


def _should_skip_dir(name: str) -> bool:
    return name in SKIP_DIR_NAMES or name.startswith(".")


def _matches_glob(name: str, glob_pattern: str) -> bool:
    pattern = (glob_pattern or "*").strip()
    if pattern == "*":
        return True
    if pattern.startswith("*."):
        return fnmatch.fnmatch(name, pattern)
    return fnmatch.fnmatch(name, pattern) or fnmatch.fnmatch(name, f"*{pattern}")


def _is_searchable_file(path: Path) -> bool:
    if not path.is_file():
        return False
    if path.suffix.lower() in TEXT_BLOCKLIST_SUFFIXES:
        return False
    return True


def _iter_search_files(
    search_root: Path,
    *,
    glob_pattern: str,
    max_files: int,
) -> list[Path]:
    """递归收集可搜索文件，上限 max_files。"""
    files: list[Path] = []
    stack: list[Path] = [search_root]

    while stack and len(files) < max_files:
        current = stack.pop()
        try:
            entries = sorted(current.iterdir(), key=lambda p: p.name)
        except OSError:
            continue

        for entry in entries:
            if len(files) >= max_files:
                break
            if entry.is_dir():
                if _should_skip_dir(entry.name):
                    continue
                stack.append(entry)
            elif entry.is_file() and _is_searchable_file(entry):
                if _matches_glob(entry.name, glob_pattern):
                    files.append(entry)

    return files


def _grep_single_file(
    file_path: Path,
    regex: re.Pattern[str],
    *,
    roots: list[Path],
    max_results: int,
    results: list[str],
) -> bool:
    """搜索单文件，返回是否已达 max_results。"""
    try:
        raw = file_path.read_bytes()
    except OSError:
        return len(results) >= max_results

    if b"\x00" in raw[:8192]:
        return len(results) >= max_results

    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError:
        try:
            text = raw.decode("utf-8-sig")
        except UnicodeDecodeError:
            return len(results) >= max_results

    rel = relative_path_from_roots(file_path, roots)
    for line_num, line in enumerate(text.splitlines(), start=1):
        if regex.search(line):
            results.append(f"{rel}:{line_num}: {line.rstrip()}")
            if len(results) >= max_results:
                return True
    return False


def grep_workspace(
    client_context: ClientContextInput | None,
    pattern: str,
    path: str = "",
    glob_pattern: str = "*",
    case_insensitive: bool = False,
    max_results: int | None = None,
    *,
    settings: Settings,
) -> str:
    """在工作区内按正则搜索代码/文本，返回 path:line: content 格式。"""
    needle = (pattern or "").strip()
    if not needle:
        return "请提供搜索 pattern（正则表达式）。"

    try:
        flags = re.IGNORECASE if case_insensitive else 0
        regex = re.compile(needle, flags)
    except re.error as exc:
        return f"无效的正则表达式：{exc}"

    roots = allowed_roots(client_context)
    if not roots:
        return "未绑定工作区或项目目录，无法搜索。"

    limit = max_results if max_results is not None else settings.agent_grep_max_results
    limit = min(max(limit, 1), settings.agent_grep_max_results)
    max_files = settings.agent_grep_max_files_scanned

    raw_path = (path or "").strip()
    files_to_search: list[Path] = []

    if raw_path:
        resolved_file, file_err = resolve_workspace_file(client_context, raw_path)
        if resolved_file is not None:
            files_to_search = [resolved_file]
        else:
            resolved_dir, dir_err = resolve_workspace_directory(client_context, raw_path)
            if resolved_dir is None:
                return file_err or dir_err or "路径无效。"
            files_to_search = _iter_search_files(
                resolved_dir,
                glob_pattern=glob_pattern,
                max_files=max_files,
            )
    else:
        for root in roots:
            files_to_search.extend(
                _iter_search_files(root, glob_pattern=glob_pattern, max_files=max_files)
            )
            if len(files_to_search) >= max_files:
                files_to_search = files_to_search[:max_files]
                break

    results: list[str] = []
    files_scanned = 0
    truncated_files = False

    for file_path in files_to_search:
        if files_scanned >= max_files:
            truncated_files = True
            break
        files_scanned += 1
        if _grep_single_file(
            file_path,
            regex,
            roots=roots,
            max_results=limit,
            results=results,
        ):
            break

    scope = normalize_relative_path(raw_path) if raw_path else "workspace"
    header = f"搜索：{needle}\n范围：{scope}\n"
    if glob_pattern and glob_pattern != "*":
        header += f"文件匹配：{glob_pattern}\n"

    if not results:
        header += "未找到匹配。\n"
        return header

    truncated_results = len(results) >= limit
    if truncated_results:
        header += f"（结果已截断，最多 {limit} 条）\n"
    if truncated_files:
        header += f"（已扫描文件数达上限 {max_files}）\n"
    header += "---\n"
    return header + "\n".join(results)


def list_workspace_directory(
    client_context: ClientContextInput | None,
    relative_dir: str = "",
    max_entries: int | None = None,
    *,
    settings: Settings,
) -> str:
    """列出工作区目录下的文件与子目录。"""
    resolved, err = resolve_workspace_directory(client_context, relative_dir)
    if resolved is None:
        return err

    roots = allowed_roots(client_context)
    cap = max_entries if max_entries is not None else settings.agent_list_dir_max_entries
    cap = min(max(cap, 1), settings.agent_list_dir_max_entries)

    rel_dir = relative_path_from_roots(resolved, roots) if roots else ""
    if not rel_dir and relative_dir.strip():
        rel_dir = normalize_relative_path(relative_dir)

    try:
        entries = sorted(resolved.iterdir(), key=lambda p: p.name.lower())
    except OSError as exc:
        return f"无法读取目录：{exc}"

    lines: list[str] = []
    truncated = False

    for entry in entries:
        if len(lines) >= cap:
            truncated = True
            break
        if entry.name.startswith("."):
            continue

        child_rel = normalize_relative_path(
            f"{rel_dir}/{entry.name}" if rel_dir else entry.name
        )
        if entry.is_dir():
            kind = "directory"
        elif entry.is_file():
            kind = "file"
        else:
            continue
        lines.append(f"{entry.name} | {kind} | {child_rel}")

    header = f"目录：{rel_dir or '.'}\n"
    header += f"条目数：{len(lines)}"
    if truncated:
        header += f"（已截断，最多 {cap} 条）"
    header += "\n---\n"

    if not lines:
        return header + "（空目录或无可见条目）\n"

    return header + "\n".join(lines)
