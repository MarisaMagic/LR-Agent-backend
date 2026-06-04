from __future__ import annotations

import json

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

from app.agent.annotation.json_utils import extract_json_object
from app.agent.annotation.schemas import ScopeParseResult

SCOPE_SYSTEM = """你是批量图片标注的范围解析助手。
根据用户自然语言和候选图片列表，选择本轮应处理的图片。
不要猜测不存在的路径。
只输出 JSON：{"selected_paths":["relative/path.jpg"],"reason":"..."}
selected_paths 必须来自候选列表的 relative_path；没有明确范围则返回空数组。"""


async def parse_image_scope(
    llm: ChatOpenAI,
    *,
    user_request: str,
    current_relative_path: str,
    candidates: list[dict],
) -> ScopeParseResult:
    prompt_candidates = candidates[:500]
    messages = [
        SystemMessage(content=SCOPE_SYSTEM),
        HumanMessage(
            content=(
                f"用户请求：{user_request}\n\n"
                f"当前文件：{current_relative_path}\n\n"
                f"图片候选：\n{json.dumps(prompt_candidates, ensure_ascii=False)}"
            )
        ),
    ]
    resp = await llm.ainvoke(messages)
    content = resp.content if hasattr(resp, "content") else str(resp)
    data = extract_json_object(str(content))
    selected = data.get("selected_paths")
    if not isinstance(selected, list):
        selected = []
    return ScopeParseResult(
        selected_paths=[str(p).strip().replace("\\", "/") for p in selected if str(p).strip()],
        reason=str(data.get("reason") or ""),
    )
