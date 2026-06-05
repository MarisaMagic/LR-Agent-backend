"""Unified per-session turn transcript for understand and batch pipelines."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

from app.agent.context_service import window_messages
from app.schemas.agent import ChatContextConfigInput, ChatMessageInput

InteractionMode = Literal["chat", "annotation"]
TurnRole = Literal["user", "assistant"]

MAX_TRANSCRIPT_CHARS = 12_000


def normalize_interaction_mode(agent_mode: str | None) -> InteractionMode:
    """Map client agent_mode to persisted message interaction_mode."""
    if agent_mode in ("annotation", "annotate"):
        return "annotation"
    return "chat"


def role_label(role: TurnRole) -> str:
    return "用户" if role == "user" else "助手"


def format_turn_line(
    *,
    role: TurnRole,
    content: str,
    interaction_mode: InteractionMode | None = None,
) -> str:
    del interaction_mode
    return f"{role_label(role)}: {content.strip()}"


class TurnLine(BaseModel):
    message_id: str
    role: TurnRole
    interaction_mode: InteractionMode | None = None
    content: str


class TurnContext(BaseModel):
    lines: list[TurnLine] = Field(default_factory=list)
    transcript: str = ""
    windowed_lines: list[TurnLine] = Field(default_factory=list)
    summary: str | None = None
    summary_up_to_message_id: str | None = None
    current_user_content: str = ""


def _truncate_transcript(transcript: str, max_chars: int = MAX_TRANSCRIPT_CHARS) -> str:
    if len(transcript) <= max_chars:
        return transcript
    lines = transcript.splitlines()
    kept: list[str] = []
    total = 0
    for line in reversed(lines):
        extra = len(line) + (1 if kept else 0)
        if total + extra > max_chars:
            break
        kept.insert(0, line)
        total += extra
    return "\n".join(kept)


def build_turn_context_from_messages(
    messages: list[ChatMessageInput],
    *,
    current_user_content: str = "",
    summary: str | None = None,
    summary_up_to_message_id: str | None = None,
    max_turns_in_window: int = 20,
    exclude_message_ids: set[str] | None = None,
) -> TurnContext:
    """Pure builder from ChatMessageInput rows (DB or client)."""
    exclude = exclude_message_ids or set()
    lines: list[TurnLine] = []
    for item in messages:
        if item.role not in ("user", "assistant"):
            continue
        mid = item.message_id or ""
        if mid and mid in exclude:
            continue
        content = item.content.strip()
        if not content:
            continue
        mode = item.interaction_mode
        lines.append(
            TurnLine(
                message_id=mid,
                role=item.role,  # type: ignore[arg-type]
                interaction_mode=mode,
                content=content,
            ),
        )

    cfg = ChatContextConfigInput(max_turns_in_window=max_turns_in_window)
    windowed_inputs = window_messages(
        [
            ChatMessageInput(
                role=line.role,
                content=line.content,
                message_id=line.message_id or None,
                interaction_mode=line.interaction_mode,
            )
            for line in lines
        ],
        cfg.max_turns_in_window,
    )
    windowed_lines = [
        TurnLine(
            message_id=m.message_id or "",
            role=m.role,  # type: ignore[arg-type]
            interaction_mode=m.interaction_mode,
            content=m.content,
        )
        for m in windowed_inputs
        if m.role in ("user", "assistant")
    ]

    transcript_lines = [
        format_turn_line(role=ln.role, content=ln.content, interaction_mode=ln.interaction_mode)
        for ln in windowed_lines
    ]
    transcript = _truncate_transcript("\n".join(transcript_lines))

    return TurnContext(
        lines=lines,
        transcript=transcript,
        windowed_lines=windowed_lines,
        summary=summary,
        summary_up_to_message_id=summary_up_to_message_id,
        current_user_content=current_user_content.strip(),
    )


def turn_context_to_chat_inputs(turn: TurnContext) -> list[ChatMessageInput]:
    """Chat history for LLM — plain role/content, no mode prefixes."""
    return [
        ChatMessageInput(
            role=line.role,
            content=line.content,
            message_id=line.message_id or None,
            interaction_mode=line.interaction_mode,
        )
        for line in turn.windowed_lines
    ]


async def build_turn_context(
    repo: object,
    session_id: str,
    *,
    user_id: object | None = None,
    current_user_content: str = "",
    up_to_message_id: str | None = None,
    exclude_message_id: str | None = None,
    max_turns_in_window: int = 20,
) -> TurnContext:
    """Load session messages from AgentChatRepository and build TurnContext."""
    import uuid

    from app.models.agent_session import AgentSession
    from app.services.agent_chat_repository import AgentChatRepository

    assert isinstance(repo, AgentChatRepository)
    exclude_ids = {exclude_message_id} if exclude_message_id else None
    messages = await repo.build_chat_message_inputs(
        session_id,
        up_to_message_id=up_to_message_id,
        exclude_message_ids=exclude_ids,
    )
    summary: str | None = None
    summary_up_to: str | None = None
    if user_id is not None:
        session_row = await repo.get_session_for_user(uuid.UUID(str(user_id)), session_id)
        if isinstance(session_row, AgentSession):
            summary = session_row.context_summary
            summary_up_to = session_row.summary_up_to_message_id
    return build_turn_context_from_messages(
        messages,
        current_user_content=current_user_content,
        summary=summary,
        summary_up_to_message_id=summary_up_to,
        max_turns_in_window=max_turns_in_window,
        exclude_message_ids=exclude_ids,
    )
