"""多轮 LLM ↔ 工具循环：stream_chunks → execute_round → 追加 ToolMessage。

拆分为两个阶段：
  1. stream_chunks: 流式产出 text/reasoning + 拦截 proposal chunks
  2. execute_round: 解析 tool_calls → 执行 SYNC / 发射 ASYNC pending
"""

import json
from collections.abc import AsyncIterator

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.tools import StructuredTool
from langchain_openai import ChatOpenAI

from app.agent.assist.proposal_streamer import ProposalStreamInterceptor
from app.agent.chat_message_builder import build_multimodal_user_message
from app.agent.stream_adapter import events_from_chunk
from app.agent.tool_dispatcher import resolve_round_tool_calls, split_resolved_calls
from app.agent.tool_invocation import ResolvedToolCall
from app.agent.tools.tool_result import format_tool_result_for_display
from app.agent.tools.workspace_file_reader import (
    VISION_TOOL_NAME,
    WRITE_TOOL_NAME,
    extract_vision_path_from_tool_result,
    extract_doc_proposal_from_tool_result,
    format_vision_tool_result_for_display,
    format_write_tool_result_for_display,
)
from app.core.config import Settings
from app.schemas.agent import StreamEventPayload


async def _try_tool_choice_retry(
    llm: ChatOpenAI,
    tools: list[StructuredTool],
    messages: list,
    user_content: str,
) -> list[ResolvedToolCall]:
    """tool_choice="any" 强制 LLM 发起 tool call（伪代码回退用）。"""
    try:
        llm_forced = llm.bind_tools(tools, tool_choice="any")
        response = await llm_forced.ainvoke(messages)
        api_calls = getattr(response, "tool_calls", None) or []
        return resolve_round_tool_calls(
            api_tool_calls=api_calls,
            response_text=str(getattr(response, "content", "") or ""),
            user_content=user_content,
        )
    except Exception:
        return []


async def _stream_tool_execution(
    *,
    tool_id: str,
    name: str,
    args: dict,
    fn_map: dict[str, object],
    provider_is_vision: bool,
    settings: Settings,
    messages: list,
    omit_file_proposal_start_delta: set[str] | None = None,
) -> AsyncIterator[StreamEventPayload]:
    """执行单个同步工具，产出 tool_start / tool_result / file_proposal* 事件。"""
    if omit_file_proposal_start_delta is None:
        omit_file_proposal_start_delta = set()
    yield StreamEventPayload(
        type="tool_start",
        tool_call_id=tool_id,
        name=name,
        arguments=json.dumps(args, ensure_ascii=False, indent=2),
    )

    fn = fn_map.get(name)
    try:
        if fn is None:
            result_text = json.dumps(
                {"ok": False, "tool": name, "status": "error", "summary": f"未知工具: {name}"},
                ensure_ascii=False,
            )
        else:
            result_text = str(fn(**args))
    except Exception as exc:
        result_text = json.dumps(
            {"ok": False, "tool": name, "status": "error", "summary": f"工具执行失败: {exc}"},
            ensure_ascii=False,
        )

    display_result = result_text
    vision_path: str | None = None
    doc_proposal: dict | None = None

    if name == VISION_TOOL_NAME:
        vision_path = extract_vision_path_from_tool_result(name, result_text)
        if vision_path:
            display_result = format_vision_tool_result_for_display(result_text)
    elif name == WRITE_TOOL_NAME:
        doc_proposal = extract_doc_proposal_from_tool_result(name, result_text)
        if doc_proposal:
            display_result = format_write_tool_result_for_display(result_text)
    else:
        display_result = format_tool_result_for_display(result_text)

    yield StreamEventPayload(
        type="tool_result",
        tool_call_id=tool_id,
        result=display_result,
    )
    messages.append(ToolMessage(content=display_result, tool_call_id=tool_id))

    if doc_proposal:
        full_content = doc_proposal["content"]
        rel_path = doc_proposal["relative_path"]
        omit_for_path = rel_path in omit_file_proposal_start_delta if omit_file_proposal_start_delta else False
        if not omit_for_path:
            yield StreamEventPayload(
                type="file_proposal_start",
                summary=doc_proposal["title"],
                image_path=rel_path,
                detail=str(len(full_content)),
            )
            chunk_size = 200
            offset = 0
            while offset < len(full_content):
                end = min(offset + chunk_size, len(full_content))
                chunk = full_content[offset:end]
                yield StreamEventPayload(
                    type="file_proposal_delta",
                    content=chunk,
                    image_path=rel_path,
                )
                offset = end
        yield StreamEventPayload(
            type="file_proposal",
            summary=doc_proposal["title"],
            content=full_content,
            image_path=rel_path,
        )

    if vision_path and provider_is_vision:
        messages.append(
            build_multimodal_user_message(
                "【附图】请根据上图回答用户关于该图片的问题。",
                image_absolute_path=vision_path,
                max_edge=settings.agent_chat_vision_max_edge,
                jpeg_quality=settings.agent_chat_vision_jpeg_quality,
            ),
        )


class ToolLoopRunner:
    """多轮 LLM ↔ 工具循环执行器。"""

    def __init__(
        self,
        llm: ChatOpenAI,
        tools: list[StructuredTool],
        fn_map: dict[str, object],
        settings: Settings,
        is_cancelled,
        user_content: str,
        provider_is_vision: bool = False,
    ) -> None:
        self.llm = llm
        self.tools = tools
        self.fn_map = fn_map
        self.settings = settings
        self.is_cancelled = is_cancelled
        self.user_content = user_content
        self.provider_is_vision = provider_is_vision
        self.vision_bootstrapped = False
        self.completed_tools: set[str] = set()
        self.tool_choice_retries = 0

    def mark_completed(self, names: set[str]) -> None:
        """标记工具为已完成（resume 时传入）。"""
        self.completed_tools |= names

    async def stream_chunks(
        self,
        messages: list,
        interceptor: ProposalStreamInterceptor,
    ) -> AsyncIterator[StreamEventPayload]:
        """流式产出 text_delta / reasoning_delta + 拦截 proposal chunks。"""
        llm_with_tools = self.llm.bind_tools(self.tools)
        _round_had_visible_output = False

        async for chunk in llm_with_tools.astream(messages):
            if await self.is_cancelled():
                return

            for event in events_from_chunk(chunk, emit_tool_chunks=False):
                if event.type in ("text_delta", "reasoning_delta") and event.content:
                    _round_had_visible_output = True
                    yield event

            for proposal_event in interceptor.on_chunk(chunk):
                yield proposal_event

            if not _round_had_visible_output and chunk.tool_call_chunks:
                first_tc_name = None
                for _tc in chunk.tool_call_chunks:
                    n = _tc.get("name")
                    if n:
                        first_tc_name = n
                        break
                if first_tc_name:
                    _round_had_visible_output = True
                    yield StreamEventPayload(
                        type="reasoning_delta",
                        content="正在分析需求，规划操作步骤…",
                    )

    async def execute_round(
        self,
        gathered: AIMessage,
        full_text: str,
        messages: list,
        interceptor: ProposalStreamInterceptor,
    ) -> AsyncIterator[StreamEventPayload]:
        """解析 tool_calls → 执行 SYNC / 发射 ASYNC pending → 产出事件。"""
        api_tool_calls = gathered.tool_calls or []

        resolved = resolve_round_tool_calls(
            api_tool_calls=api_tool_calls,
            completed_tools=frozenset(self.completed_tools),
        )

        if not resolved:
            return  # 无 tool call，本轮结束

        messages.append(gathered)
        split = split_resolved_calls(resolved)
        streamed_paths = interceptor.collected_paths()

        for call in split.immediate:
            if await self.is_cancelled():
                return
            async for event in _stream_tool_execution(
                tool_id=call.tool_call_id,
                name=call.name,
                args=call.arguments,
                fn_map=self.fn_map,
                provider_is_vision=self.provider_is_vision,
                settings=self.settings,
                messages=messages,
                omit_file_proposal_start_delta=streamed_paths,
            ):
                yield event
            if call.name == VISION_TOOL_NAME:
                self.vision_bootstrapped = True

        if split.async_pending:
            from app.agent.assist.pending_emitter import emit_tool_pending

            async for event in emit_tool_pending(split.async_pending):
                yield event
