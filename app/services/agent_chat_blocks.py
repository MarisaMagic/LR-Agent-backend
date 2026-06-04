from typing import Any

from app.schemas.agent import StreamEventPayload

PIPELINE_STAGE_LABELS: dict[str, str] = {
    "prepare": "准备批量标注",
    "task": "解析任务",
    "catalog": "扫描图片",
    "scope": "解析范围",
    "plan": "生成计划",
    "workers": "批量处理",
    "worker": "处理图片",
    "tool": "工具调用",
}


def _label_for_pipeline_stage(stage: str) -> str:
    return PIPELINE_STAGE_LABELS.get(stage, stage)


def _apply_annotation_progress_to_blocks(
    blocks: list[dict[str, Any]],
    event: StreamEventPayload,
) -> list[dict[str, Any]]:
    next_blocks = [dict(block) for block in blocks]
    incoming = {
        "stage": event.stage or "",
        "label": _label_for_pipeline_stage(event.stage or ""),
        "message": event.message or "",
        "status": event.status or "running",
        "detail": event.detail,
    }

    def upsert_steps(steps: list[dict[str, Any]]) -> list[dict[str, Any]]:
        updated: list[dict[str, Any]] = []
        for step in steps:
            if (
                step.get("status") == "running"
                and step.get("stage") != event.stage
                and event.stage != "worker"
            ):
                updated.append({**step, "status": "done"})
            else:
                updated.append(dict(step))
        if event.stage == "worker":
            idx = next(
                (i for i, s in enumerate(updated) if s.get("stage") == "worker" and s.get("message") == event.message),
                -1,
            )
            if idx >= 0:
                updated[idx] = incoming
                return updated
            trimmed = [s for s in updated if s.get("stage") != "worker"] + [incoming]
            workers = [s for s in trimmed if s.get("stage") == "worker"]
            rest = [s for s in trimmed if s.get("stage") != "worker"]
            return rest + workers[-12:]
        idx = next((i for i, s in enumerate(updated) if s.get("stage") == event.stage), -1)
        if idx >= 0:
            updated[idx] = incoming
            return updated
        return updated + [incoming]

    pipeline_idx = next((i for i, b in enumerate(next_blocks) if b.get("type") == "annotation_pipeline"), -1)
    if pipeline_idx < 0:
        next_blocks.append(
            {
                "type": "annotation_pipeline",
                "collapsed": False,
                "steps": [incoming],
            },
        )
        return next_blocks

    block = next_blocks[pipeline_idx]
    next_blocks[pipeline_idx] = {
        **block,
        "steps": upsert_steps(list(block.get("steps") or [])),
    }
    return next_blocks


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

    if event.type == "annotation_progress" and event.stage:
        return _apply_annotation_progress_to_blocks(next_blocks, event)

    if event.type == "annotation_proposal" and event.proposal is not None:
        for i, block in enumerate(next_blocks):
            if block.get("type") == "annotation_pipeline":
                next_blocks[i] = {
                    **block,
                    "collapsed": True,
                    "steps": [
                        {**step, "status": "done" if step.get("status") == "running" else step.get("status")}
                        for step in (block.get("steps") or [])
                    ],
                }
        proposal_block = {
            "type": "annotation_proposal",
            "proposal": event.proposal,
            "status": "pending",
        }
        existing_idx = next((i for i, b in enumerate(next_blocks) if b.get("type") == "annotation_proposal"), -1)
        if existing_idx >= 0:
            next_blocks[existing_idx] = proposal_block
        else:
            next_blocks.append(proposal_block)
        return next_blocks

    return next_blocks


def collapse_assistant_blocks(blocks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for block in blocks:
        if block.get("type") in ("reasoning", "tool_call", "annotation_pipeline"):
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


def blocks_to_preview(blocks: list[dict[str, Any]], *, max_len: int = 128) -> str:
    text = blocks_to_text(blocks)
    if text.strip():
        line = " ".join(text.strip().split())
        return f"{line[:max_len]}…" if len(line) > max_len else line

    for block in blocks:
        if block.get("type") == "annotation_proposal":
            proposal = block.get("proposal") if isinstance(block.get("proposal"), dict) else {}
            stats = proposal.get("stats") if isinstance(proposal.get("stats"), dict) else {}
            succeeded = int(stats.get("succeeded") or 0)
            total_boxes = int(stats.get("totalBoxes") or stats.get("total_boxes") or 0)
            summary = str(proposal.get("summary") or "").strip()
            if summary:
                line = " ".join(summary.split())
                return f"{line[:max_len]}…" if len(line) > max_len else line
            if succeeded:
                hint = f"批量标注 {succeeded} 张"
                if total_boxes:
                    hint += f" · {total_boxes} 框"
                return hint
    return ""
