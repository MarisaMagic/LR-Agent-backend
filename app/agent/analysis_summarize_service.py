"""数据分析结果流式解读。"""

from __future__ import annotations

from collections.abc import AsyncIterator

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

from app.schemas.agent import StreamEventPayload

ANALYSIS_SUMMARIZE_SYSTEM = """你是 LR-Agent 数据分析解读助手。根据用户请求、脚本说明与脚本标准输出（stdout），用中文 Markdown 写出分析结论。

要求：
- 直接回答用户问题，突出关键发现；可用小标题、列表或表格
- 所有数字必须来自 stdout，勿编造
- 不要粘贴完整 Python 脚本
- 若 stdout 为 JSON，先理解再用人话总结
"""


async def stream_analysis_summary(
    llm: ChatOpenAI,
    *,
    user_request: str,
    explanation: str,
    script: str,
    stdout: str,
    conversation_transcript: str = "",
) -> AsyncIterator[StreamEventPayload]:
    transcript = conversation_transcript.strip() or "（无历史）"
    stdout_trimmed = (stdout or "").strip()[:24_000]
    script_trimmed = (script or "").strip()[:8_000]
    human = (
        f"【对话上下文】\n{transcript}\n\n"
        f"【用户请求】\n{user_request.strip()}\n\n"
        f"【脚本说明】\n{(explanation or '').strip() or '（无）'}\n\n"
        f"【脚本摘要】\n{script_trimmed[:1200]}{'…' if len(script_trimmed) > 1200 else ''}\n\n"
        f"【脚本 stdout】\n{stdout_trimmed or '（无输出）'}\n"
    )
    messages = [
        SystemMessage(content=ANALYSIS_SUMMARIZE_SYSTEM),
        HumanMessage(content=human),
    ]
    async for chunk in llm.astream(messages):
        content = chunk.content
        if isinstance(content, str) and content:
            yield StreamEventPayload(type="text_delta", content=content)
        elif isinstance(content, list):
            for part in content:
                if isinstance(part, dict) and part.get("type") == "text":
                    text = str(part.get("text") or "")
                    if text:
                        yield StreamEventPayload(type="text_delta", content=text)
