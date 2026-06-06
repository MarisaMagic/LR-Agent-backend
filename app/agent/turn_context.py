"""轮次上下文：从会话消息构建统一的对话转录与窗口化历史。

供下游两类场景复用：
  - orchestrator：裁剪对话窗口 → 生成 transcript 供回合理解 → 转回 ChatMessageInput
  - API 端点（agent / annotation_agent）：从仓储加载会话后提取 transcript
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

from app.agent.context_service import window_messages
from app.schemas.agent import ChatContextConfigInput, ChatMessageInput

InteractionMode = Literal["chat", "annotation"]
TurnRole = Literal["user", "assistant"]

# 转录文本上限，防止回合理解 prompt 过长
MAX_TRANSCRIPT_CHARS = 12_000


def normalize_interaction_mode(agent_mode: str | None) -> InteractionMode:
    """将客户端 agent_mode 映射为持久化的 interaction_mode。"""
    if agent_mode == "annotation":
        return "annotation"
    return "chat"


def role_label(role: TurnRole) -> str:
    return "用户" if role == "user" else "助手"


def format_turn_line(
    *,
    role: TurnRole,
    content: str,
) -> str:
    """格式化单行对话，用于组装 transcript。"""
    return f"{role_label(role)}: {content.strip()}"


class TurnLine(BaseModel):
    """单条对话记录（含 message_id 与 interaction_mode 元数据）。"""
    message_id: str
    role: TurnRole
    interaction_mode: InteractionMode | None = None
    content: str


class TurnContext(BaseModel):
    """一轮对话的上下文快照。

    - lines：全量有效消息
    - windowed_lines / transcript：裁剪窗口后的消息与「用户/助手」转录文本
    - summary 字段：来自会话级历史摘要（由 orchestrator 另行注入 LLM 消息链）
    """
    lines: list[TurnLine] = Field(default_factory=list)
    transcript: str = ""
    windowed_lines: list[TurnLine] = Field(default_factory=list)
    summary: str | None = None
    summary_up_to_message_id: str | None = None
    current_user_content: str = ""


def _truncate_transcript(transcript: str, max_chars: int = MAX_TRANSCRIPT_CHARS) -> str:
    """从尾部保留完整行，截断超长 transcript。"""
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
    """从消息列表（DB 或请求）纯函数构建 TurnContext，无 I/O。"""
    exclude = exclude_message_ids or set()
    lines: list[TurnLine] = []
    for item in messages:
        if item.role not in ("user", "assistant"):
            continue
        mid = item.message_id or ""
        # 排除占位中的助手消息（orchestrator 传入 assistant_message_id）
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
        format_turn_line(role=ln.role, content=ln.content)
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
    """将窗口化历史转回 ChatMessageInput，供 build_lc_messages 使用（不含模式前缀）。"""
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
    """从 AgentChatRepository 加载会话消息并构建 TurnContext（API 端点使用）。"""
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
