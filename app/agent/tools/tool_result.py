"""统一 ToolMessage JSON 结构。"""

from __future__ import annotations

import json
from typing import Any


def build_tool_result(
    *,
    ok: bool,
    tool: str,
    status: str,
    summary: str,
    **extra: Any,
) -> str:
    payload: dict[str, Any] = {
        "ok": ok,
        "tool": tool,
        "status": status,
        "summary": summary,
        "file_written": extra.pop("file_written", False),
        "proposal_pending": extra.pop("proposal_pending", False),
    }
    payload.update(extra)
    return json.dumps(payload, ensure_ascii=False)


def format_tool_result_for_display(result_text: str) -> str:
    """将 ToolMessage JSON 格式化为 UI/模型可读摘要。"""
    try:
        data = json.loads(result_text)
    except json.JSONDecodeError:
        return result_text
    if not isinstance(data, dict):
        return result_text
    if data.get("summary"):
        return str(data["summary"])
    if data.get("message"):
        return str(data["message"])
    return result_text


def parse_tool_result(result_text: str) -> dict[str, Any] | None:
    try:
        data = json.loads(result_text)
    except json.JSONDecodeError:
        return None
    return data if isinstance(data, dict) else None
