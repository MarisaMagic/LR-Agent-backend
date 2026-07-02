"""Vision bootstrap path resolution and tool execution helpers for assist chat."""

from __future__ import annotations

import json
import uuid
from collections.abc import AsyncIterator
from typing import Callable

from langchain_core.messages import ToolMessage

from app.agent.chat_message_builder import build_multimodal_user_message
from app.agent.tools.workspace_file_reader import (
    IMAGE_SUFFIXES,
    VISION_TOOL_NAME,
    extract_vision_path_from_tool_result,
    format_vision_tool_result_for_display,
)
from app.agent.tools.workspace_path import resolve_workspace_file
from app.core.config import Settings
from app.schemas.agent import ClientContextInput, StreamEventPayload


def pick_vision_relative_path(
    client_context: ClientContextInput | None,
) -> str:
    """Best relative path for auto vision load (active_relative_path > empty for UI file)."""
    if client_context is not None:
        rel = (client_context.active_relative_path or "").strip()
        if rel:
            return rel
    return ""


def vision_relative_from_user_text(user_content: str) -> str:
    """Extract data/foo.jpg style path from user message when understanding is missing."""
    import re

    match = re.search(
        r"(?:^|[\s「『])([\w./\\-]+\.(?:jpg|jpeg|png|gif|webp|bmp|ico))\b",
        user_content,
        re.IGNORECASE,
    )
    if not match:
        return ""
    return match.group(1).replace("\\", "/")


def can_bootstrap_vision(
    client_context: ClientContextInput | None,
    relative_path: str,
) -> bool:
    resolved, _err = resolve_workspace_file(client_context, relative_path)
    if resolved is None:
        return False
    return resolved.suffix.lower() in IMAGE_SUFFIXES


async def stream_vision_tool_execution(
    *,
    tool_id: str,
    relative_path: str,
    vision_fn: Callable[..., str],
    provider_is_vision: bool,
    settings: Settings,
    messages: list,
) -> AsyncIterator[StreamEventPayload]:
    """Run read_image_for_vision, yield tool events, append ToolMessage + optional image."""
    args = {"relative_path": relative_path}
    yield StreamEventPayload(
        type="tool_start",
        tool_call_id=tool_id,
        name=VISION_TOOL_NAME,
        arguments=json.dumps(args, ensure_ascii=False, indent=2),
    )

    try:
        result_text = str(vision_fn(**args))
    except Exception as exc:
        result_text = f"工具执行失败: {exc}"

    display_result = result_text
    vision_path = extract_vision_path_from_tool_result(VISION_TOOL_NAME, result_text)
    if vision_path:
        display_result = format_vision_tool_result_for_display(result_text)

    yield StreamEventPayload(
        type="tool_result",
        tool_call_id=tool_id,
        result=display_result,
    )
    messages.append(ToolMessage(content=display_result, tool_call_id=tool_id))

    if vision_path and provider_is_vision:
        messages.append(
            build_multimodal_user_message(
                "【附图】请根据上图回答用户关于该图片的问题。",
                image_absolute_path=vision_path,
                max_edge=settings.agent_chat_vision_max_edge,
                jpeg_quality=settings.agent_chat_vision_jpeg_quality,
            ),
        )
