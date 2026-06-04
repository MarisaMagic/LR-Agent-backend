"""Agent-driven image scope resolution via tools (no regex routing)."""
from __future__ import annotations

import json
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import StructuredTool
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

from app.agent.annotation.schemas import ScopeParseResult

SCOPE_AGENT_SYSTEM = """你是批量图片标注的范围解析 Agent。

用户已选择 Annotation 执行模式。你必须根据用户自然语言，从项目图片目录候选中选出本轮要处理的图片。

工作流程：
1. 可先调用 list_project_images 按文件夹、文件名关键词浏览候选（支持模糊查询）。
2. 确定范围后，必须调用 select_annotation_images 提交最终列表（relative_path 只能来自候选）。

规则：
- 用户提到多张图（如 7.jpg 和 8.jpg）须全部选中。
- 用户说某文件夹下所有图，可 list 后 select 该批路径。
- 无法确定时 select 空列表并在 reason 说明。
- 不要编造不在候选中的路径。"""


class ListImagesInput(BaseModel):
    parent_folder: str = Field(
        default="",
        description="按父目录过滤，如 data；空表示不限",
    )
    name_contains: str = Field(
        default="",
        description="文件名包含的子串，如 7.jpg 或 7",
    )
    limit: int = Field(default=80, ge=1, le=200)


class SelectImagesInput(BaseModel):
    selected_paths: list[str] = Field(
        default_factory=list,
        description="最终选中的 relative_path 列表",
    )
    reason: str = Field(default="", description="选择理由")


def _build_catalog_maps(
    candidates: list[dict[str, Any]],
) -> tuple[dict[str, dict[str, Any]], list[dict[str, Any]]]:
    by_path: dict[str, dict[str, Any]] = {}
    for item in candidates:
        rel = str(item.get("relative_path") or "").strip().replace("\\", "/")
        if rel:
            by_path[rel] = item
    return by_path, candidates


def _tool_list_images(
    candidates: list[dict[str, Any]],
    parent_folder: str = "",
    name_contains: str = "",
    limit: int = 80,
) -> str:
    parent = parent_folder.strip().replace("\\", "/").strip("/")
    needle = name_contains.strip().lower()
    matched: list[dict[str, Any]] = []
    for item in candidates:
        rel = str(item.get("relative_path") or "").replace("\\", "/")
        name = str(item.get("name") or "").lower()
        par = str(item.get("parent") or "").replace("\\", "/")
        if parent and par != parent and not rel.startswith(f"{parent}/"):
            continue
        if needle and needle not in name.lower() and needle not in rel.lower():
            continue
        matched.append(
            {
                "relative_path": rel,
                "name": item.get("name"),
                "parent": par,
                "index": item.get("index"),
            }
        )
        if len(matched) >= limit:
            break
    return json.dumps(
        {"count": len(matched), "images": matched},
        ensure_ascii=False,
    )


def _tool_select_images(
    by_path: dict[str, dict[str, Any]],
    selected_paths: list[str],
    reason: str = "",
) -> str:
    valid: list[str] = []
    invalid: list[str] = []
    for raw in selected_paths:
        key = str(raw).strip().replace("\\", "/")
        if not key:
            continue
        if key in by_path:
            valid.append(key)
        else:
            invalid.append(key)
    return json.dumps(
        {
            "ok": len(invalid) == 0,
            "selected_paths": valid,
            "invalid_paths": invalid,
            "reason": reason,
        },
        ensure_ascii=False,
    )


async def resolve_scope_with_agent(
    llm: ChatOpenAI,
    *,
    user_request: str,
    current_relative_path: str,
    candidates: list[dict[str, Any]],
    max_rounds: int = 6,
) -> ScopeParseResult:
    by_path, catalog = _build_catalog_maps(candidates)
    if not catalog:
        return ScopeParseResult(selected_paths=[], reason="项目目录无图片候选")

    def list_project_images(
        parent_folder: str = "",
        name_contains: str = "",
        limit: int = 80,
    ) -> str:
        return _tool_list_images(catalog, parent_folder, name_contains, limit)

    def select_annotation_images(
        selected_paths: list[str],
        reason: str = "",
    ) -> str:
        return _tool_select_images(by_path, selected_paths, reason)

    tools = [
        StructuredTool.from_function(
            func=list_project_images,
            name="list_project_images",
            description="浏览/筛选项目内图片候选，按文件夹或文件名关键词过滤",
            args_schema=ListImagesInput,
        ),
        StructuredTool.from_function(
            func=select_annotation_images,
            name="select_annotation_images",
            description="确认本轮要批量标注的图片相对路径列表",
            args_schema=SelectImagesInput,
        ),
    ]

    llm_tools = llm.bind_tools(tools)
    fn_map = {t.name: t for t in tools}
    messages = [
        SystemMessage(content=SCOPE_AGENT_SYSTEM),
        HumanMessage(
            content=(
                f"用户请求：{user_request}\n\n"
                f"当前打开文件：{current_relative_path or '（无）'}\n\n"
                f"候选图片总数：{len(catalog)}"
            )
        ),
    ]

    final_paths: list[str] = []
    final_reason = ""

    for _ in range(max_rounds):
        response = await llm_tools.ainvoke(messages)
        messages.append(response)

        tool_calls = getattr(response, "tool_calls", None) or []
        if not tool_calls:
            break

        for call in tool_calls:
            name = call.get("name") or ""
            args = call.get("args") or {}
            if not isinstance(args, dict):
                args = {}
            tool_id = call.get("id") or "scope-tool"
            fn = fn_map.get(name)
            try:
                if fn is None:
                    result_text = f"未知工具: {name}"
                else:
                    result_text = str(fn.invoke(args))
            except Exception as exc:
                result_text = f"工具执行失败: {exc}"

            if name == "select_annotation_images":
                try:
                    payload = json.loads(result_text)
                    final_paths = list(payload.get("selected_paths") or [])
                    final_reason = str(payload.get("reason") or "")
                except json.JSONDecodeError:
                    final_reason = result_text

            messages.append(
                ToolMessage(content=result_text, tool_call_id=tool_id),
            )

        if final_paths or any(
            call.get("name") == "select_annotation_images" for call in tool_calls
        ):
            break

    if final_paths:
        return ScopeParseResult(selected_paths=final_paths, reason=final_reason)

    return ScopeParseResult(
        selected_paths=[],
        reason=final_reason or "Agent 未确认图片范围，请更具体说明文件夹或文件名",
    )
