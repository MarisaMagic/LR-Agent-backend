from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from app.core.config import Settings
from app.schemas.agent import ChatContextConfigInput, ChatContextInput, ChatMessageInput, ChatStreamRequest

CHAT_SYSTEM_PROMPT = """在 LR-Agent 系统内回答用户问题。结合【你的身份】中的模型信息作答，勿自称独立产品助手或其它未配置的模型。"""

ASSIST_SYSTEM_PROMPT = """你正在 LR-Agent 中协助用户，必要时可使用只读工具查询账户信息、应用说明与当前界面上下文（工作区路径、打开文件、标注项目等）。"""

SUMMARIZE_PROMPT = """请将以下对话历史压缩为简洁中文摘要，保留关键事实、用户目标与已达成结论。
只输出摘要正文，不要加标题或前后缀。"""


def estimate_tokens(text: str) -> int:
    return max(1, len(text) // 4)


def _config(req: ChatStreamRequest, settings: Settings) -> ChatContextConfigInput:
    if req.context and req.context.config:
        return req.context.config
    return ChatContextConfigInput(
        max_context_tokens=settings.agent_default_max_context_tokens,
        reserve_completion_tokens=settings.agent_default_reserve_completion_tokens,
        max_turns_in_window=settings.agent_default_max_turns_in_window,
        summarize_trigger_ratio=settings.agent_default_summarize_trigger_ratio,
        min_turns_before_summarize=settings.agent_default_min_turns_before_summarize,
    )


def window_messages(
    messages: list[ChatMessageInput],
    max_turns: int,
) -> list[ChatMessageInput]:
    if max_turns <= 0:
        return messages
    max_messages = max_turns * 2
    if len(messages) <= max_messages:
        return messages
    return messages[-max_messages:]


def build_lc_messages(
    req: ChatStreamRequest,
    settings: Settings,
    *,
    system_prompt: str = CHAT_SYSTEM_PROMPT,
    last_user_image_absolute_path: str | None = None,
    attach_vision_to_last_user: bool = False,
) -> tuple[list, int, bool]:
    """Return (lc_messages, token_estimate, needs_summarize)."""
    cfg = _config(req, settings)
    msgs = list(req.messages)
    if req.user_content.strip():
        last = msgs[-1] if msgs else None
        if not (last and last.role == "user" and last.content.strip() == req.user_content.strip()):
            msgs = [*msgs, ChatMessageInput(role="user", content=req.user_content)]

    windowed = window_messages(msgs, cfg.max_turns_in_window)

    lc_messages: list = [SystemMessage(content=system_prompt)]
    if req.context and req.context.summary:
        lc_messages.append(
            SystemMessage(content=f"【此前对话摘要】\n{req.context.summary}"),
        )

    from app.agent.chat_message_builder import build_multimodal_user_message

    for idx, item in enumerate(windowed):
        is_last = idx == len(windowed) - 1
        if item.role == "user":
            if (
                attach_vision_to_last_user
                and is_last
                and last_user_image_absolute_path
            ):
                lc_messages.append(
                    build_multimodal_user_message(
                        item.content,
                        image_absolute_path=last_user_image_absolute_path,
                        max_edge=settings.agent_chat_vision_max_edge,
                        jpeg_quality=settings.agent_chat_vision_jpeg_quality,
                    ),
                )
            else:
                lc_messages.append(HumanMessage(content=item.content))
        elif item.role == "assistant":
            lc_messages.append(AIMessage(content=item.content))
        elif item.role == "system":
            lc_messages.append(SystemMessage(content=item.content))

    token_estimate = sum(estimate_tokens(m.content) for m in msgs)
    if req.context and req.context.summary:
        token_estimate += estimate_tokens(req.context.summary)

    budget = cfg.max_context_tokens - cfg.reserve_completion_tokens
    turn_pairs = sum(1 for m in msgs if m.role == "user")
    needs_summarize = (
        token_estimate >= int(budget * cfg.summarize_trigger_ratio)
        and turn_pairs >= cfg.min_turns_before_summarize
        and not (req.context and req.context.summary)
    )

    return lc_messages, token_estimate, needs_summarize


def messages_for_summary(messages: list[ChatMessageInput]) -> str:
    lines: list[str] = []
    for item in messages:
        prefix = "用户" if item.role == "user" else "助手"
        lines.append(f"{prefix}: {item.content}")
    return "\n".join(lines)
