"""Unicode normalization for LLM-generated text."""

from __future__ import annotations

from typing import Any


def sanitize_unicode_text(text: str) -> str:
    """Remove lone UTF-16 surrogates that break downstream parsers."""
    if not text:
        return ""
    text = text.encode("utf-8", "surrogatepass").decode("utf-8", "replace")
    return text.replace("\ufffd", "")


def sanitize_json_value(value: Any) -> Any:
    """递归清洗快照/JSON 中的字符串，避免沙箱 UTF-8 写盘失败。"""
    if isinstance(value, str):
        return sanitize_unicode_text(value)
    if isinstance(value, dict):
        return {
            sanitize_unicode_text(k) if isinstance(k, str) else k: sanitize_json_value(v)
            for k, v in value.items()
        }
    if isinstance(value, list):
        return [sanitize_json_value(item) for item in value]
    if isinstance(value, tuple):
        return tuple(sanitize_json_value(item) for item in value)
    return value
