"""标注质量报告 LLM 撰写。"""

from __future__ import annotations

import json
from collections.abc import AsyncIterator

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

from app.schemas.agent import StreamEventPayload

QUALITY_REPORT_COMPOSE_SYSTEM = """你是 LR-Agent 标注质量报告撰写助手。根据提供的标注质量指标、一致性检查发现和图表路径，用中文 Markdown 撰写结构化质量报告。

要求：
- 固定章节：概览、覆盖率、标签分布、一致性问题、风险文件清单、改进建议
- 所有数字必须来自提供的 metrics / findings JSON，禁止编造
- 在合适章节嵌入图表，使用给定 chart_paths 中的相对路径，格式：![标题](charts/xxx.png)
- 对 critical / warning 级别发现给出明确复核建议
- 语言简洁专业，适合交付给标注团队或甲方
- 不要输出代码块包裹整篇报告
"""


async def stream_quality_report_compose(
    llm: ChatOpenAI,
    *,
    compose_payload: dict,
) -> AsyncIterator[StreamEventPayload]:
    payload_json = json.dumps(compose_payload, ensure_ascii=False, indent=2)
    if len(payload_json) > 48_000:
        payload_json = payload_json[:48_000] + "\n…（已截断）"

    human = (
        "【质量指标与发现 JSON】\n"
        f"{payload_json}\n\n"
        "请基于以上数据撰写完整 Markdown 质量报告。"
    )
    messages = [
        SystemMessage(content=QUALITY_REPORT_COMPOSE_SYSTEM),
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
