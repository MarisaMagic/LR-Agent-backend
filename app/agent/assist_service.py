import json
from collections.abc import AsyncIterator

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.tools import StructuredTool
from langchain_openai import ChatOpenAI

from app.agent.stream_adapter import events_from_chunk
from app.agent.tools.registry import tool_fn_map
from app.schemas.agent import StreamEventPayload


async def stream_assist(
    llm: ChatOpenAI,
    lc_messages: list,
    tools: list[StructuredTool],
    *,
    max_tool_rounds: int,
    is_cancelled,
) -> AsyncIterator[StreamEventPayload]:
    yield StreamEventPayload(type="preparing", stage="streaming")
    llm_with_tools = llm.bind_tools(tools)
    fn_map = tool_fn_map(tools)
    messages = list(lc_messages)
    open_tool_ids: dict[str, str] = {}

    for _ in range(max_tool_rounds + 1):
        if await is_cancelled():
            return

        gathered: AIMessage | None = None
        async for chunk in llm_with_tools.astream(messages):
            if await is_cancelled():
                return
            for event in events_from_chunk(chunk):
                if event.type == "tool_start" and event.tool_call_id:
                    open_tool_ids[event.tool_call_id] = event.name or "tool"
                yield event
            if gathered is None:
                gathered = chunk
            else:
                gathered = gathered + chunk

        if gathered is None:
            break

        tool_calls = gathered.tool_calls or []
        if not tool_calls:
            break

        messages.append(gathered)
        for call in tool_calls:
            if await is_cancelled():
                return
            tool_id = call.get("id") or "tool-unknown"
            name = call.get("name") or "tool"
            args = call.get("args") or {}
            if not isinstance(args, dict):
                try:
                    args = json.loads(args) if args else {}
                except json.JSONDecodeError:
                    args = {}

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

            yield StreamEventPayload(
                type="tool_result",
                tool_call_id=tool_id,
                result=result_text,
            )
            messages.append(
                ToolMessage(content=result_text, tool_call_id=tool_id),
            )
