"""对话上下文服务：将请求消息组装为 LangChain 消息链，并评估是否需要压缩摘要。

职责：
  - 裁剪对话窗口、拼接系统提示词与历史摘要
  - 估算 token 用量，判断是否需要触发 summarize
  - 为摘要压缩任务格式化对话文本
"""

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from app.core.config import Settings
from app.schemas.agent import ChatContextConfigInput, ChatContextInput, ChatMessageInput, ChatStreamRequest

# 纯 chat 模式默认系统提示词（assist 模式由 context_snapshot 组装）
CHAT_SYSTEM_PROMPT = """在 LR-Agent 系统内回答用户问题。结合【你的身份】中的模型信息作答，勿自称独立产品助手或其它未配置的模型。"""

SUMMARIZE_PROMPT = """请将以下对话历史压缩为简洁中文摘要，保留关键事实、用户目标与已达成结论。
只输出摘要正文，不要加标题或前后缀。"""


def estimate_tokens(text: str) -> int:
    """按字符数粗估 token（len // 4），用于预算判断而非精确计费。"""
    return max(1, len(text) // 4)


def _config(req: ChatStreamRequest, settings: Settings) -> ChatContextConfigInput:
    """合并请求级与默认上下文配置（窗口大小、token 预算、摘要触发阈值等）。"""
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
    """按轮次裁剪消息列表，保留最近 max_turns 轮（每轮 user + assistant）。"""
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
) -> tuple[list, int, bool]:
    """构建 LangChain 消息链，返回 (lc_messages, token_estimate, needs_summarize)。"""
    cfg = _config(req, settings)
    msgs = list(req.messages)

    # 确保当前用户输入已纳入消息列表（避免与最后一条 user 消息重复）
    if req.user_content.strip():
        last = msgs[-1] if msgs else None
        if not (last and last.role == "user" and last.content.strip() == req.user_content.strip()):
            msgs = [*msgs, ChatMessageInput(role="user", content=req.user_content)]

    windowed = window_messages(msgs, cfg.max_turns_in_window)

    # 系统提示词 + 可选历史摘要
    lc_messages: list = [SystemMessage(content=system_prompt)]
    if req.context and req.context.summary:
        lc_messages.append(
            SystemMessage(content=f"【此前对话摘要】\n{req.context.summary}"),
        )

    for item in windowed:
        if item.role == "user":
            lc_messages.append(HumanMessage(content=item.content))
        elif item.role == "assistant":
            lc_messages.append(AIMessage(content=item.content))
        elif item.role == "system":
            lc_messages.append(SystemMessage(content=item.content))

    # 基于全量消息（非窗口裁剪后）估算 token，用于摘要触发判断
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


def append_client_tool_results_to_messages(
    lc_messages: list,
    client_tool_results: list,
    *,
    user_content: str = "",
) -> list:
    """Resume 时在消息链末尾追加 AIMessage(tool_calls) + ToolMessage 对（支持累积多轮）。"""
    if not client_tool_results:
        return lc_messages

    from langchain_core.messages import AIMessage, ToolMessage

    for ctr in client_tool_results:
        args: dict = {"user_request": user_content.strip()}
        try:
            import json

            parsed = json.loads(ctr.result)
            if isinstance(parsed, dict) and parsed.get("user_request"):
                args["user_request"] = str(parsed["user_request"])
        except Exception:
            pass
        lc_messages.append(
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "id": ctr.tool_call_id,
                        "name": ctr.name,
                        "args": args,
                    },
                ],
            ),
        )
        lc_messages.append(
            ToolMessage(content=ctr.result, tool_call_id=ctr.tool_call_id),
        )
    return lc_messages


def messages_for_summary(messages: list[ChatMessageInput]) -> str:
    """将消息列表格式化为「用户/助手」对话文本，供摘要 LLM 使用。"""
    lines: list[str] = []
    for item in messages:
        prefix = "用户" if item.role == "user" else "助手"
        lines.append(f"{prefix}: {item.content}")
    return "\n".join(lines)
