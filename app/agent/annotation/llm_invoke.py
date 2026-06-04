"""LLM invoke helpers without OpenAI json_schema (DeepSeek-compatible)."""
from __future__ import annotations

from typing import TypeVar

from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from pydantic import BaseModel

from app.agent.annotation.json_utils import extract_json_object

T = TypeVar("T", bound=BaseModel)


async def invoke_json_model(
    llm: ChatOpenAI,
    messages: list[BaseMessage],
    model_cls: type[T],
    *,
    extra_instruction: str = "",
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
    data = extract_json_object(str(content))
    return model_cls.model_validate(data)
