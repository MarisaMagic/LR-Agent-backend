"""统一工具调度：按 registry runner 拆分 API 调用结果。"""

from __future__ import annotations

from dataclasses import dataclass

from app.agent.tool_invocation import ResolvedToolCall, normalize_api_tool_calls
from app.agent.tools.tool_registry_meta import ToolRunner, get_tool_runner


@dataclass(frozen=True)
class SplitToolCalls:
    immediate: list[ResolvedToolCall]
    async_pending: list[ResolvedToolCall]


def split_resolved_calls(calls: list[ResolvedToolCall]) -> SplitToolCalls:
    immediate: list[ResolvedToolCall] = []
    async_pending: list[ResolvedToolCall] = []
    for call in calls:
        if get_tool_runner(call.name) is ToolRunner.ASYNC:
            async_pending.append(call)
        else:
            immediate.append(call)
    return SplitToolCalls(immediate=immediate, async_pending=async_pending)


def resolve_round_tool_calls(
    *,
    api_tool_calls: list,
    completed_tools: frozenset[str] | None = None,
) -> list[ResolvedToolCall]:
    """仅使用 LLM API tool_calls 解析，不再回退伪代码。"""
    return normalize_api_tool_calls(
        api_tool_calls,
        completed_tools=completed_tools,
    )
