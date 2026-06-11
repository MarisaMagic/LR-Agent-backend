"""加载会话对话 transcript，供 prepare API 与回合理解复用。"""

from __future__ import annotations

import uuid
from typing import Any

from app.agent.turn_context import build_turn_context


async def load_conversation_transcript(
    repo: Any,
    session_id: str | None,
    *,
    user_id: uuid.UUID,
    user_request: str = "",
    max_turns_in_window: int | None = None,
) -> str:
    if not session_id or not str(session_id).strip():
        return "（无历史）"
    kwargs: dict[str, Any] = {
        "user_id": user_id,
        "current_user_content": user_request,
    }
    if max_turns_in_window is not None:
        kwargs["max_turns_in_window"] = max_turns_in_window
    turn = await build_turn_context(repo, session_id, **kwargs)
    transcript = (turn.transcript or "").strip()
    return transcript or "（无历史）"
