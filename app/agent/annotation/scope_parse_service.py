"""Single-shot batch image scope parsing (fusion-style, no multi-turn tools)."""
from __future__ import annotations

import json

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

from app.agent.annotation.json_utils import extract_json_object
from app.agent.annotation.schemas import ScopeParseResult

SCOPE_SINGLE_SHOT_SYSTEM = """你是批量图片标注的范围解析助手。

根据用户自然语言与候选图片列表，选出本轮应处理的图片。
只输出 JSON：{"selected_paths":["relative/path.jpg"],"reason":"..."}
规则：
- selected_paths 必须来自候选 relative_path，可多张（如 7.jpg 和 8.jpg 都要选）
- 不要编造不在候选中的路径
- 无法确定时返回空数组并在 reason 说明"""


async def parse_batch_scope_single_shot(
    llm: ChatOpenAI,
    *,
    user_request: str,
    current_relative_path: str,
    candidates: list[dict],
    max_candidates_in_prompt: int = 500,
) -> ScopeParseResult:
    if not candidates:
        return ScopeParseResult(selected_paths=[], reason="项目目录无图片候选")

    by_path: dict[str, dict] = {}
    for item in candidates:
        rel = str(item.get("relative_path") or "").strip().replace("\\", "/")
        if rel:
            by_path[rel] = item

    prompt_candidates = [
        {
            "relative_path": c.get("relative_path"),
            "name": c.get("name"),
            "parent": c.get("parent"),
        }
        for c in candidates[:max_candidates_in_prompt]
    ]

    messages = [
        SystemMessage(content=SCOPE_SINGLE_SHOT_SYSTEM),
        HumanMessage(
            content=(
                f"用户请求：{user_request}\n\n"
                f"当前打开文件：{current_relative_path or '（无）'}\n\n"
                f"候选图片（{len(prompt_candidates)} 项）：\n"
                f"{json.dumps(prompt_candidates, ensure_ascii=False)}"
            )
        ),
    ]
    resp = await llm.ainvoke(messages)
    content = resp.content if hasattr(resp, "content") else str(resp)
    data = extract_json_object(str(content))
    raw_paths = data.get("selected_paths")
    selected: list[str] = []
    if isinstance(raw_paths, list):
        for raw in raw_paths:
            key = str(raw).strip().replace("\\", "/")
            if key and key in by_path:
                selected.append(key)
    reason = str(data.get("reason") or "")
    if not selected:
        return ScopeParseResult(
            selected_paths=[],
            reason=reason or "未从候选中解析到图片，请更具体说明文件夹或文件名",
        )
    return ScopeParseResult(selected_paths=selected, reason=reason)
