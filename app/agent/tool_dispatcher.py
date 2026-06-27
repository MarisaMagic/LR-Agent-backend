"""统一工具调度：按 registry runner 拆分 API / 伪代码解析结果。"""

from __future__ import annotations

from dataclasses import dataclass

from app.agent.tool_invocation import ResolvedToolCall, resolve_tool_calls
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
    response_text: str,
    user_content: str,
    completed_tools: frozenset[str] | None = None,
    enable_pseudo_parsing: bool = True,
) -> list[ResolvedToolCall]:
    return resolve_tool_calls(
        api_tool_calls=api_tool_calls,
        response_text=response_text,
        user_content=user_content,
        completed_tools=completed_tools,
        enable_pseudo_parsing=enable_pseudo_parsing,
    )
