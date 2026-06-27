"""Assist 模式流式推理：统一 ToolDispatcher 驱动的多轮 LLM ↔ 工具循环。"""

import json
import uuid
from collections.abc import AsyncIterator

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.tools import StructuredTool
from langchain_openai import ChatOpenAI

from app.agent.assist_vision import (
    can_bootstrap_vision,
    pick_vision_relative_path,
    stream_vision_tool_execution,
    vision_relative_from_user_text,
)
from app.agent.chat_message_builder import build_multimodal_user_message
from app.agent.context_service import append_client_tool_results_to_messages
from app.agent.stream_adapter import events_from_chunk
from app.agent.tool_dispatcher import resolve_round_tool_calls, split_resolved_calls
from app.agent.tool_invocation import ResolvedToolCall, tool_mentioned_in_text
from app.agent.tools.registry import tool_fn_map
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
from app.schemas.agent import ClientContextInput, ClientToolCallPayload, StreamEventPayload


def _normalize_tool_call(call: dict) -> tuple[str, str, dict]:
    tool_id = str(call.get("id") or f"tool-{uuid.uuid4().hex[:12]}")
    name = str(call.get("name") or "tool")
    args = call.get("args") or {}
    if not isinstance(args, dict):
        try:
            args = json.loads(args) if args else {}
        except json.JSONDecodeError:
            args = {}
    return tool_id, name, args


def _extract_json_string(buf: str, key: str) -> tuple[str | None, bool]:
    """从可能不完整的 JSON 缓冲区中提取字符串字段值。

    处理标准 JSON 转义（\\n \\t \\r \\\\ \\"），返回 (解码后的字符串, 是否已闭合)。
    若 key 尚未出现则返回 (None, False)；
    若出现但字符串值未闭合引号则返回 (已累积部分, False)；
    若字符串值已闭合引号则返回 (完整值, True)。
    """
    idx = buf.find(f'"{key}"')
    if idx == -1:
        return None, False
    try:
        colon = buf.index(":", idx)
        quote = buf.index('"', colon + 1)
    except ValueError:
        return None, False
    result: list[str] = []
    i = quote + 1
    while i < len(buf):
        c = buf[i]
        if c == "\\" and i + 1 < len(buf):
            nxt = buf[i + 1]
            esc: dict[str, str] = {"n": "\n", "t": "\t", "r": "\r", "\\": "\\", '"': '"'}
            result.append(esc.get(nxt, nxt))
            i += 2
        elif c == '"':
            return "".join(result), True
        else:
            result.append(c)
            i += 1
    # 缓冲区末尾字符串未闭合——返回已累积部分，继续等待后续 chunk
    return ("".join(result) if result else None), False


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
    """omit_file_proposal_start_delta 是一个 relative_path 集合，
    表示 astream 拦截器已经发送过这些文件的 file_proposal_start/delta 流式事件。"""
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
                {
                    "ok": False,
                    "tool": name,
                    "status": "error",
                    "summary": f"未知工具: {name}",
                },
                ensure_ascii=False,
            )
        else:
            result_text = str(fn(**args))
    except Exception as exc:
        result_text = json.dumps(
            {
                "ok": False,
                "tool": name,
                "status": "error",
                "summary": f"工具执行失败: {exc}",
            },
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


async def _emit_tool_pending(
    calls: list[ResolvedToolCall],
) -> AsyncIterator[StreamEventPayload]:
    """发出 tool_pending SSE（兼容 client_tool_pending）并结束当前 HTTP 轮次。"""
    pending_calls = [
        ClientToolCallPayload(
            tool_call_id=c.tool_call_id,
            name=c.name,
            arguments=c.arguments,
        )
        for c in calls
    ]
    for c in calls:
        yield StreamEventPayload(
            type="tool_start",
            tool_call_id=c.tool_call_id,
            name=c.name,
            arguments=json.dumps(c.arguments, ensure_ascii=False, indent=2),
        )
    yield StreamEventPayload(
        type="tool_pending",
        client_tool_calls=pending_calls,
    )
    yield StreamEventPayload(
        type="client_tool_pending",
        client_tool_calls=pending_calls,
    )


async def _try_tool_choice_retry(
    llm: ChatOpenAI,
    tools: list[StructuredTool],
    messages: list,
    user_content: str,
) -> list[ResolvedToolCall]:
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
    completed_tools: set[str] = set()

    if client_tool_results:
        for ctr in client_tool_results:
            completed_tools.add(ctr.name)
        append_client_tool_results_to_messages(
            messages,
            client_tool_results,
            user_content=user_content,
        )

    tool_choice_retries = 0

    wants_vision = bool(vision_relative_from_user_text(user_content))
    if (
        client_context
        and client_context.turn_understanding is not None
        and not client_context.turn_understanding.needs_vision_input
    ):
        wants_vision = False

    vision_bootstrapped = False
    if not is_resume and provider_is_vision and vision_fn is not None and wants_vision:
        rel = pick_vision_relative_path(client_context, None)
        if not rel and user_content.strip():
            rel = vision_relative_from_user_text(user_content)
        if can_bootstrap_vision(client_context, rel):
            tool_id = f"lr-vision-bootstrap-{uuid.uuid4().hex[:10]}"
            async for event in stream_vision_tool_execution(
                tool_id=tool_id,
                relative_path=rel,
                vision_fn=vision_fn,  # type: ignore[arg-type]
                provider_is_vision=provider_is_vision,
                settings=settings,
                messages=messages,
            ):
                yield event
            vision_bootstrapped = True

    for round_idx in range(max_tool_rounds + 1):
        if await is_cancelled():
            return

        gathered: AIMessage | None = None
        pending_text: list[str] = []
        pending_reasoning: list[str] = []

        # 追踪本轮是否已输出文本/推理；用于在 tool call 生成阶段发送占位反馈
        _round_had_visible_output = False

        # file_proposal 流式状态（在 astream 中实时拦截 write_workspace_file 的 tool_call_chunks）
        # 改为按 tc_index 索引的 dict，支持同一轮内多个 write_workspace_file 调用
        fp_states: dict[int, dict] = {}

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
                    pending_reasoning.append(event.content)
                    yield event

            # ── 拦截 write_workspace_file 的 tool_call_chunks，实时流式推送 file_proposal ──
            for tc in chunk.tool_call_chunks or []:
                tc_name = tc.get("name")
                tc_args = tc.get("args") or ""
                tc_idx = tc.get("index")

                if tc_idx is None:
                    continue

                if tc_name == WRITE_TOOL_NAME:
                    fp_states.setdefault(tc_idx, {
                        "args_buf": "",
                        "title_sent": False,
                        "content_sent_len": 0,
                        "rel_path": "",
                    })
                    fp_states[tc_idx]["args_buf"] = tc_args
                elif tc_idx in fp_states and tc_args:
                    state = fp_states[tc_idx]
                    # 处理 cumulative（全量替换）或 incremental（追加）
                    if tc_args.startswith(state["args_buf"]):
                        state["args_buf"] = tc_args
                    else:
                        state["args_buf"] += tc_args

                if tc_idx in fp_states and fp_states[tc_idx]["args_buf"]:
                    state = fp_states[tc_idx]
                    # 尝试提取 relative_path → file_proposal_start（仅当值已闭合时发送）
                    if not state["title_sent"]:
                        rel_path, rel_closed = _extract_json_string(state["args_buf"], "relative_path")
                        if rel_path and rel_closed:
                            state["title_sent"] = True
                            state["rel_path"] = rel_path
                            yield StreamEventPayload(
                                type="file_proposal_start",
                                summary=rel_path,
                                image_path=rel_path,
                                detail="0",
                            )

                    # 尝试提取 content → file_proposal_delta
                    content, _ = _extract_json_string(state["args_buf"], "content")
                    if content is not None and len(content) > state["content_sent_len"]:
                        delta = content[state["content_sent_len"]:]
                        state["content_sent_len"] = len(content)
                        if delta:
                            yield StreamEventPayload(
                                type="file_proposal_delta",
                                content=delta,
                                image_path=state.get("rel_path"),
                            )

            # ── 无 reasoning/text 输出时，首次检测到 tool call chunk 名称时发出占位推理反馈 ──
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
        api_tool_calls = gathered.tool_calls or []

        resolved = resolve_round_tool_calls(
            api_tool_calls=api_tool_calls,
            response_text=full_text,
            user_content=user_content,
            completed_tools=frozenset(completed_tools),
            enable_pseudo_parsing=settings.agent_assist_pseudo_tool_parsing,
        )

        if not resolved and tool_mentioned_in_text(full_text):
            if tool_choice_retries < 2:
                tool_choice_retries += 1
                if full_text.strip():
                    messages.append(AIMessage(content=full_text))
                mentioned = tool_mentioned_in_text(full_text) or "tool"
                messages.append(
                    HumanMessage(
                        content=(
                            f"【系统】请使用 tool call 调用 {mentioned}，"
                            "不要在正文中写伪代码。"
                            "写文件请用 write_workspace_file(relative_path, content)。"
                        ),
                    ),
                )
                forced = await _try_tool_choice_retry(llm, tools, messages, user_content)
                if forced:
                    resolved = forced
                else:
                    continue

        if resolved:
            split = split_resolved_calls(resolved)
            messages.append(gathered)

            # 收集 interceptor 已发送流式事件的 relative_path 集合
            streamed_paths: set[str] = set()
            for state in fp_states.values():
                args_buf = state.get("args_buf", "")
                rel_path, rel_closed = _extract_json_string(args_buf, "relative_path")
                if rel_path and rel_closed and state.get("title_sent"):
                    streamed_paths.add(rel_path)

            for call in split.immediate:
                if await is_cancelled():
                    return
                async for event in _stream_tool_execution(
                    tool_id=call.tool_call_id,
                    name=call.name,
                    args=call.arguments,
                    fn_map=fn_map,
                    provider_is_vision=provider_is_vision,
                    settings=settings,
                    messages=messages,
                    omit_file_proposal_start_delta=streamed_paths,
                ):
                    yield event
                if call.name == VISION_TOOL_NAME:
                    vision_bootstrapped = True

            if split.async_pending:
                async for event in _emit_tool_pending(split.async_pending):
                    yield event
                return

            continue

        if (
            round_idx == 0
            and not is_resume
            and wants_vision
            and provider_is_vision
            and not vision_bootstrapped
            and vision_fn is not None
        ):
            rel = pick_vision_relative_path(client_context, None)
            if not rel and user_content.strip():
                rel = vision_relative_from_user_text(user_content)
            if can_bootstrap_vision(client_context, rel):
                tool_id = f"lr-vision-fallback-{uuid.uuid4().hex[:10]}"
                async for event in stream_vision_tool_execution(
                    tool_id=tool_id,
                    relative_path=rel,
                    vision_fn=vision_fn,  # type: ignore[arg-type]
                    provider_is_vision=provider_is_vision,
                    settings=settings,
                    messages=messages,
                ):
                    yield event
                vision_bootstrapped = True
                continue

        if tool_mentioned_in_text(full_text) and round_idx < max_tool_rounds:
            messages.append(AIMessage(content=full_text))
            messages.append(
                HumanMessage(
                    content="【系统】检测到未执行的工具伪代码。请发起真实 tool call 后再总结。",
                ),
            )
            continue

        break
