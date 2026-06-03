from typing import Any

from app.schemas.agent import StreamEventPayload


def apply_stream_event_to_blocks(
    blocks: list[dict[str, Any]],
    event: StreamEventPayload,
) -> list[dict[str, Any]]:
    next_blocks = [dict(block) for block in blocks]

    if event.type == "text_delta" and event.content:
        if next_blocks and next_blocks[-1].get("type") == "text":
            last = next_blocks[-1]
            next_blocks[-1] = {
                **last,
                "content": str(last.get("content", "")) + event.content,
            }
            return next_blocks
        next_blocks.append({"type": "text", "content": event.content})
        return next_blocks

    if event.type == "reasoning_delta" and event.content:
        idx = next(
            (i for i, block in enumerate(next_blocks) if block.get("type") == "reasoning"),
            -1,
        )
        if idx >= 0:
            block = next_blocks[idx]
            next_blocks[idx] = {
                **block,
                "content": str(block.get("content", "")) + event.content,
            }
            return next_blocks
        next_blocks.append(
            {
                "type": "reasoning",
                "content": event.content,
                "collapsed": False,
            },
        )
        return next_blocks

    if event.type == "tool_start" and event.tool_call_id:
        existing_idx = next(
            (
                i
                for i, block in enumerate(next_blocks)
                if block.get("type") == "tool_call" and block.get("id") == event.tool_call_id
            ),
            -1,
        )
        if existing_idx >= 0:
            block = next_blocks[existing_idx]
            next_blocks[existing_idx] = {
                **block,
                "arguments": str(block.get("arguments", "")) + str(event.arguments or ""),
            }
            return next_blocks
        next_blocks.append(
            {
                "type": "tool_call",
                "id": event.tool_call_id,
                "name": event.name or "tool",
                "arguments": event.arguments or "",
                "status": "running",
                "collapsed": False,
            },
        )
        return next_blocks

    if event.type == "tool_result" and event.tool_call_id:
        idx = next(
            (
                i
                for i, block in enumerate(next_blocks)
                if block.get("type") == "tool_call" and block.get("id") == event.tool_call_id
            ),
            -1,
        )
        if idx >= 0:
            block = next_blocks[idx]
            next_blocks[idx] = {
                **block,
                "result": event.result or "",
                "status": "done",
                "collapsed": True,
            }
        return next_blocks

    return next_blocks


def collapse_assistant_blocks(blocks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for block in blocks:
        if block.get("type") in ("reasoning", "tool_call"):
            result.append({**block, "collapsed": True})
        else:
            result.append(dict(block))
    return result


def blocks_to_text(blocks: list[dict[str, Any]]) -> str:
    parts: list[str] = []
    for block in blocks:
        if block.get("type") == "text":
            content = str(block.get("content", "")).strip()
            if content:
                parts.append(content)
    return "\n".join(parts)
