"""Assist 模式流式推理：在工具增强对话中执行多轮 LLM ↔ 工具循环。

由 orchestrator 在 assist 路由下调用，流程概览：
  1. 绑定只读工具 → 按需预加载视觉图片（bootstrap）
  2. 多轮循环：LLM 流式推理 → 解析 tool_calls → 执行工具 → 回填 ToolMessage
  3. 无工具调用时输出最终文本；首轮可触发视觉 fallback
"""

import json
import uuid
from collections.abc import AsyncIterator

from langchain_core.messages import AIMessage, ToolMessage
from langchain_core.tools import StructuredTool
from langchain_openai import ChatOpenAI

from app.agent.assist_vision import (
    can_bootstrap_vision,
    pick_vision_relative_path,
    stream_vision_tool_execution,
    vision_relative_from_user_text,
)
from app.agent.chat_message_builder import build_multimodal_user_message
from app.agent.stream_adapter import events_from_chunk
from app.agent.tools.registry import tool_fn_map
from app.agent.tools.workspace_file_reader import (
    VISION_TOOL_NAME,
    extract_vision_path_from_tool_result,
    format_vision_tool_result_for_display,
)
from app.core.config import Settings
from app.schemas.agent import ClientContextInput, StreamEventPayload
from app.agent.turn_understanding_service import TurnUnderstandingResult


def _normalize_tool_call(call: dict) -> tuple[str, str, dict]:
    """将 LLM 返回的 tool_call 规范化为 (id, name, args)。"""
    tool_id = str(call.get("id") or f"tool-{uuid.uuid4().hex[:12]}")
    name = str(call.get("name") or "tool")
    args = call.get("args") or {}
    if not isinstance(args, dict):
        try:
            args = json.loads(args) if args else {}
        except json.JSONDecodeError:
            args = {}
    return tool_id, name, args


async def _stream_tool_execution(
    *,
    tool_id: str,
    name: str,
    args: dict,
    fn_map: dict[str, object],
    provider_is_vision: bool,
    settings: Settings,
    messages: list,
) -> AsyncIterator[StreamEventPayload]:
    """执行单个工具，产出 tool_start / tool_result 事件，并将结果追加到消息列表。"""
    yield StreamEventPayload(
        type="tool_start",
        tool_call_id=tool_id,
        name=name,
        arguments=json.dumps(args, ensure_ascii=False, indent=2),
    )

    fn = fn_map.get(name)
    try:
        if fn is None:
            result_text = f"未知工具: {name}"
        else:
            result_text = str(fn(**args))
    except Exception as exc:
        result_text = f"工具执行失败: {exc}"

    display_result = result_text
    vision_path: str | None = None
    if name == VISION_TOOL_NAME:
        vision_path = extract_vision_path_from_tool_result(name, result_text)
        if vision_path:
            display_result = format_vision_tool_result_for_display(result_text)

    yield StreamEventPayload(
        type="tool_result",
        tool_call_id=tool_id,
        result=display_result,
    )
    messages.append(ToolMessage(content=display_result, tool_call_id=tool_id))

    # 视觉工具成功后，向消息链注入多模态用户消息供下一轮 LLM 看图
    if vision_path and provider_is_vision:
        messages.append(
            build_multimodal_user_message(
                "【附图】请根据上图回答用户关于该图片的问题。",
                image_absolute_path=vision_path,
                max_edge=settings.agent_chat_vision_max_edge,
                jpeg_quality=settings.agent_chat_vision_jpeg_quality,
            ),
        )


async def stream_assist(
    llm: ChatOpenAI,
    lc_messages: list,
    tools: list[StructuredTool],
    *,
    settings: Settings,
    max_tool_rounds: int,
    is_cancelled,
    provider_is_vision: bool = False,
    needs_vision_input: bool = False,
    client_context: ClientContextInput | None = None,
    understanding: TurnUnderstandingResult | None = None,
    user_content: str = "",
) -> AsyncIterator[StreamEventPayload]:
    """Assist 模式主入口：多轮工具调用循环，直至 LLM 产出最终回复或无剩余轮次。"""
    yield StreamEventPayload(type="preparing", stage="streaming")
    llm_with_tools = llm.bind_tools(tools)
    fn_map = tool_fn_map(tools)
    messages = list(lc_messages)
    vision_fn = fn_map.get(VISION_TOOL_NAME)

    wants_vision = needs_vision_input or bool(vision_relative_from_user_text(user_content))

    # ── 阶段 1：视觉预加载（bootstrap）────────────────────────────────────
    # 在首轮 LLM 调用前主动加载图片，避免模型未主动调用视觉工具
    vision_bootstrapped = False
    if provider_is_vision and vision_fn is not None and wants_vision:
        rel = pick_vision_relative_path(client_context, understanding)
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

    # ── 阶段 2：多轮 LLM ↔ 工具循环 ─────────────────────────────────────
    for round_idx in range(max_tool_rounds + 1):
        if await is_cancelled():
            return

        gathered: AIMessage | None = None
        pending_text: list[str] = []
        pending_reasoning: list[str] = []

        # 流式收集本轮 LLM 输出；工具调用阶段暂不向外推送文本增量
        async for chunk in llm_with_tools.astream(messages):
            if await is_cancelled():
                return
            for event in events_from_chunk(chunk, emit_tool_chunks=False):
                if event.type == "text_delta" and event.content:
                    pending_text.append(event.content)
                elif event.type == "reasoning_delta" and event.content:
                    pending_reasoning.append(event.content)
            if gathered is None:
                gathered = chunk
            else:
                gathered = gathered + chunk

        if gathered is None:
            break

        tool_calls = gathered.tool_calls or []

        if tool_calls:
            # LLM 请求工具：依次执行，结果回填后继续下一轮
            messages.append(gathered)
            for call in tool_calls:
                if await is_cancelled():
                    return
                tool_id, name, args = _normalize_tool_call(call)
                async for event in _stream_tool_execution(
                    tool_id=tool_id,
                    name=name,
                    args=args,
                    fn_map=fn_map,
                    provider_is_vision=provider_is_vision,
                    settings=settings,
                    messages=messages,
                ):
                    yield event
                if name == VISION_TOOL_NAME:
                    vision_bootstrapped = True
            continue

        # ── 阶段 3：视觉 fallback ─────────────────────────────────────────
        # 首轮无 tool_calls 但用户需要看图时，主动补一次视觉工具调用
        if (
            round_idx == 0
            and wants_vision
            and provider_is_vision
            and not vision_bootstrapped
            and vision_fn is not None
        ):
            rel = pick_vision_relative_path(client_context, understanding)
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

        # ── 阶段 4：输出最终回复 ───────────────────────────────────────────
        for part in pending_reasoning:
            yield StreamEventPayload(type="reasoning_delta", content=part)
        for part in pending_text:
            yield StreamEventPayload(type="text_delta", content=part)
        break
