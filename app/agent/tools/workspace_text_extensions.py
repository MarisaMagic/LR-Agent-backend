"""工作区可写文本/代码文件后缀白名单（与前端 workspaceTextExtensions 保持一致）。"""

from __future__ import annotations

ALLOWED_TEXT_FILE_EXTENSIONS: frozenset[str] = frozenset(
    {
        ".md",
        ".txt",
        ".json",
        ".yaml",
        ".yml",
        ".csv",
        ".tsv",
        ".xml",
        ".html",
        ".htm",
        ".rst",
        ".py",
        ".js",
        ".ts",
        ".tsx",
        ".jsx",
        ".cpp",
        ".cc",
        ".cxx",
        ".c",
        ".h",
        ".hpp",
        ".cs",
        ".java",
        ".go",
        ".rs",
        ".sql",
        ".sh",
        ".bat",
        ".ps1",
        ".toml",
        ".ini",
        ".cfg",
        ".env",
    }
)


def is_allowed_text_extension(ext: str) -> bool:
    return ext.lower() in ALLOWED_TEXT_FILE_EXTENSIONS
