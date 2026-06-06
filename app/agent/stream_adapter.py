from langchain_core.messages import AIMessageChunk

from app.schemas.agent import StreamEventPayload


def events_from_chunk(
    chunk: AIMessageChunk,
    *,
    emit_tool_chunks: bool = True,
) -> list[StreamEventPayload]:
    events: list[StreamEventPayload] = []
    content = chunk.content
    if isinstance(content, str) and content:
        events.append(StreamEventPayload(type="text_delta", content=content))
    elif isinstance(content, list):
        for part in content:
            if isinstance(part, dict) and part.get("type") == "text":
                text = part.get("text", "")
                if text:
                    events.append(StreamEventPayload(type="text_delta", content=text))

    additional = getattr(chunk, "additional_kwargs", None) or {}
    reasoning = additional.get("reasoning_content")
    if reasoning:
        events.append(StreamEventPayload(type="reasoning_delta", content=str(reasoning)))

    if not emit_tool_chunks:
        return events

    tool_chunks = chunk.tool_call_chunks or []
    for tool_chunk in tool_chunks:
        tool_id = tool_chunk.get("id") or tool_chunk.get("index")
        name = tool_chunk.get("name") or "tool"
        args = tool_chunk.get("args") or ""
        if isinstance(args, dict):
            import json

            args = json.dumps(args, ensure_ascii=False)
        events.append(
            StreamEventPayload(
                type="tool_start",
                tool_call_id=str(tool_id) if tool_id is not None else "tool-pending",
                name=str(name),
                arguments=str(args),
            ),
        )

    return events
