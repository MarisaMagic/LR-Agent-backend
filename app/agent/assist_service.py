"""Assist 模式流式推理：编排 ToolLoopRunner + VisionAutoLoader + ProposalStreamInterceptor。"""

from collections.abc import AsyncIterator

from langchain_core.messages import AIMessage
from langchain_core.tools import StructuredTool
from langchain_openai import ChatOpenAI

from app.agent.assist.proposal_streamer import ProposalStreamInterceptor
from app.agent.assist.tool_loop import ToolLoopRunner
from app.agent.assist.vision_bootstrap import VisionAutoLoader
from app.agent.context_service import append_client_tool_results_to_messages
from app.agent.stream_adapter import events_from_chunk
from app.agent.tools.registry import tool_fn_map
from app.agent.tools.workspace_file_reader import VISION_TOOL_NAME
from app.core.config import Settings
from app.schemas.agent import ClientContextInput, StreamEventPayload


async def stream_assist(
    llm: ChatOpenAI,
    lc_messages: list,
    tools: list[StructuredTool],
    *,
    settings: Settings,
    max_tool_rounds: int,
    is_cancelled,
    provider_is_vision: bool = False,
    client_context: ClientContextInput | None = None,
    user_content: str = "",
    client_tool_results: list | None = None,
) -> AsyncIterator[StreamEventPayload]:
    yield StreamEventPayload(type="preparing", stage="streaming")

    llm_with_tools = llm.bind_tools(tools)
    fn_map = tool_fn_map(tools)
    messages = list(lc_messages)
    vision_fn = fn_map.get(VISION_TOOL_NAME)
    is_resume = bool(client_tool_results)

    # ── 子模块实例化 ─────────────────────────────────────────────────
    vision = VisionAutoLoader(
        vision_fn=vision_fn,
        provider_is_vision=provider_is_vision,
        settings=settings,
        client_context=client_context,
        user_content=user_content,
    )
    interceptor = ProposalStreamInterceptor()
    loop = ToolLoopRunner(
        llm=llm,
        tools=tools,
        fn_map=fn_map,
        settings=settings,
        is_cancelled=is_cancelled,
        user_content=user_content,
        provider_is_vision=provider_is_vision,
    )

    # ── resume：注入已完成的工具结果 ─────────────────────────────────
    if client_tool_results:
        completed_names = {ctr.name for ctr in client_tool_results}
        loop.mark_completed(completed_names)
        append_client_tool_results_to_messages(
            messages, client_tool_results, user_content=user_content
        )

    # ── 视觉预加载（首轮） ───────────────────────────────────────────
    if vision.should_load(is_resume):
        async for event in vision.try_bootstrap(messages):
            yield event
        loop.vision_bootstrapped = True

    # ── 主循环 ───────────────────────────────────────────────────────
    for round_idx in range(max_tool_rounds + 1):
        if await is_cancelled():
            return

        gathered: AIMessage | None = None
        pending_text: list[str] = []
        _round_had_visible_output = False

        # 单次 astream：同时收集 text/reasoning + 拦截 proposal + 累加 gathered
        async for chunk in llm_with_tools.astream(messages):
            if await is_cancelled():
                return

            for event in events_from_chunk(chunk, emit_tool_chunks=False):
                if event.type == "text_delta" and event.content:
                    _round_had_visible_output = True
                    pending_text.append(event.content)
                    yield event
                elif event.type == "reasoning_delta" and event.content:
                    _round_had_visible_output = True
                    yield event

            # 提案流式拦截
            for proposal_event in interceptor.on_chunk(chunk):
                yield proposal_event

            # 占位推理反馈
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

            if gathered is None:
                gathered = chunk
            else:
                gathered = gathered + chunk

        if gathered is None:
            break

        full_text = "".join(pending_text)

        # 执行本轮 tool calls（由 ToolLoopRunner 统一解析 + 执行 + pending）
        had_tool_call = False
        async for event in loop.execute_round(
            gathered=gathered,
            full_text=full_text,
            messages=messages,
            interceptor=interceptor,
        ):
            had_tool_call = True
            if event.type == "tool_pending":
                yield event
                return
            yield event

        if had_tool_call:
            continue

        # 无 tool call 时的处理
        # 视觉 fallback（仅第一轮）
        if (
            round_idx == 0
            and not is_resume
            and vision.should_load(is_resume)
            and not loop.vision_bootstrapped
        ):
            async for event in vision.try_fallback(messages):
                yield event
            loop.vision_bootstrapped = True
            continue

        break
