"""工作区文本文件策略：二进制/富媒体黑名单 + 默认文本可写。

须与 LR-Agent-frontend/src/shared/workspaceTextExtensions.ts 保持同步。
"""

from __future__ import annotations

# 禁止 write_workspace_file / 文本编辑的扩展名
TEXT_WRITE_BLOCKLIST: frozenset[str] = frozenset(
    {
        ".png",
        ".jpg",
        ".jpeg",
        ".gif",
        ".webp",
        ".bmp",
        ".ico",
        ".svg",
        ".pdf",
        ".docx",
        ".doc",
        ".zip",
        ".rar",
        ".7z",
        ".exe",
        ".dll",
        ".so",
        ".dylib",
        ".mp3",
        ".mp4",
        ".avi",
        ".mov",
        ".woff",
        ".woff2",
        ".ttf",
        ".otf",
    }
)

# read_workspace_file 禁止当纯文本读取的格式（与写盘黑名单一致）
TEXT_BLOCKLIST_SUFFIXES = TEXT_WRITE_BLOCKLIST


def is_blocked_text_extension(ext: str) -> bool:
    normalized = ext.lower() if ext.startswith(".") else f".{ext.lower()}"
    return normalized in TEXT_WRITE_BLOCKLIST


def is_allowed_text_extension(ext: str) -> bool:
    """非黑名单扩展名均可写（兼容旧调用方）。"""
    return not is_blocked_text_extension(ext)
