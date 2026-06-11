"""LLM invoke helpers without OpenAI json_schema (DeepSeek-compatible)."""
from __future__ import annotations

import logging
from typing import Any, TypeVar

from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from pydantic import BaseModel

from app.agent.annotation.json_utils import extract_json_object

logger = logging.getLogger(__name__)

T = TypeVar("T", bound=BaseModel)

_LOG_TRUNCATE = 4_000


def coerce_null_string_fields(data: dict[str, Any], *fields: str) -> dict[str, Any]:
    """将指定字段的 JSON null 归一为 \"\"，避免 LLM 误用 null 导致 Pydantic 校验失败。"""
    if not fields:
        return data
    out = dict(data)
    for key in fields:
        if key in out and out[key] is None:
            out[key] = ""
    return out


async def invoke_json_model(
    llm: ChatOpenAI,
    messages: list[BaseMessage],
    model_cls: type[T],
    *,
    extra_instruction: str = "",
    log_label: str | None = None,
    null_string_fields: tuple[str, ...] = (),
) -> T:
    sys_parts = [m.content for m in messages if isinstance(m, SystemMessage)]
    human_parts = [m.content for m in messages if isinstance(m, HumanMessage)]
    prompt = [
        *(sys_parts or ["你是结构化 JSON 助手。"]),
        extra_instruction,
        "只输出一个 JSON 对象，不要 markdown 代码块。",
        "\n".join(human_parts),
    ]
    merged = [
        SystemMessage(content="\n".join(p for p in prompt[:2] if p)),
        HumanMessage(content="\n".join(prompt[2:])),
    ]
    resp = await llm.ainvoke(merged)
    content = resp.content if hasattr(resp, "content") else str(resp)
    raw_text = str(content)
    data = extract_json_object(raw_text)
    if null_string_fields:
        data = coerce_null_string_fields(data, *null_string_fields)
    if log_label:
        logger.info(
            "[%s] llm raw response (truncated): %s",
            log_label,
            raw_text[:_LOG_TRUNCATE],
        )
        logger.info("[%s] extracted json: %s", log_label, data)
    return model_cls.model_validate(data)
