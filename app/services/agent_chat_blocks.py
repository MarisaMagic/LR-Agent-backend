from typing import Any

from app.schemas.agent import StreamEventPayload

PIPELINE_IMAGE_DETAIL_STAGES = frozenset({"worker", "judge", "retry"})
PIPELINE_RUNNING_DETAIL_TAIL = 5


def _is_image_detail_stage(stage: str | None) -> bool:
    return stage in PIPELINE_IMAGE_DETAIL_STAGES


def _extract_image_path_from_worker_message(message: str | None) -> str | None:
    if not message:
        return None
    trimmed = message.strip()
    colon = trimmed.find("：")
    if colon < 0:
        return None
    path = trimmed[colon + 1 :].strip()
    return path or None


def _resolve_pipeline_image_path(step: dict[str, Any], incoming_path: str | None = None) -> str | None:
    explicit = (incoming_path or step.get("imagePath") or step.get("image_path") or "").strip()
    if explicit:
        return explicit
    return _extract_image_path_from_worker_message(str(step.get("message") or ""))


def _trim_running_detail_tail(steps: list[dict[str, Any]]) -> list[dict[str, Any]]:
    main = [s for s in steps if not _is_image_detail_stage(s.get("stage"))]
    details = [s for s in steps if _is_image_detail_stage(s.get("stage"))]
    terminal = [s for s in details if s.get("status") != "running"]
    running = [s for s in details if s.get("status") == "running"]
    return main + terminal + running[-PIPELINE_RUNNING_DETAIL_TAIL:]


def _upsert_image_detail_step(
    steps: list[dict[str, Any]],
    incoming: dict[str, Any],
) -> list[dict[str, Any]]:
    image_path = _resolve_pipeline_image_path(incoming, incoming.get("imagePath") or incoming.get("image_path"))
    with_path = {**incoming}
    if image_path:
        with_path["imagePath"] = image_path

    main = [dict(s) for s in steps if not _is_image_detail_stage(s.get("stage"))]
    details = [dict(s) for s in steps if _is_image_detail_stage(s.get("stage"))]

    if image_path:
        idx = next(
            (i for i, s in enumerate(details) if _resolve_pipeline_image_path(s) == image_path),
            -1,
        )
        if idx >= 0:
            details[idx] = with_path
        else:
            details.append(with_path)
    else:
        idx = next(
            (
                i
                for i, s in enumerate(details)
                if s.get("stage") == incoming.get("stage") and s.get("message") == incoming.get("message")
            ),
            -1,
        )
        if idx >= 0:
            details[idx] = with_path
        else:
            details.append(with_path)

    return _trim_running_detail_tail(main + details)


PIPELINE_STAGE_LABELS: dict[str, str] = {
    "prepare": "准备批量标注",
    "task": "解析任务",
    "catalog": "扫描图片",
    "scope": "解析范围",
    "plan": "生成计划",
    "workers": "批量处理",
    "worker": "处理图片",
    "judge": "评分复核",
    "retry": "重新打标签",
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
        "imagePath": event.image_path,
    }

    def upsert_steps(steps: list[dict[str, Any]]) -> list[dict[str, Any]]:
        updated: list[dict[str, Any]] = []
        for step in steps:
            if (
                step.get("status") == "running"
                and step.get("stage") != event.stage
                and not _is_image_detail_stage(event.stage)
            ):
                updated.append({**step, "status": "done"})
            else:
                updated.append(dict(step))
        if _is_image_detail_stage(event.stage):
            return _upsert_image_detail_step(updated, incoming)
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
            existing_status = next_blocks[existing_idx].get("status")
            if existing_status:
                proposal_block["status"] = existing_status
            next_blocks = [b for b in next_blocks if b.get("type") != "annotation_proposal"]
        next_blocks.append(proposal_block)
        return next_blocks

    return next_blocks


def finalize_pipeline_steps_in_blocks(
    blocks: list[dict[str, Any]],
    *,
    terminal_status: str = "done",
) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for block in blocks:
        if block.get("type") != "annotation_pipeline":
            result.append(dict(block))
            continue
        steps = [
            {
                **step,
                "status": terminal_status
                if step.get("status") == "running"
                else step.get("status"),
            }
            for step in (block.get("steps") or [])
        ]
        result.append({**block, "steps": steps})
    return result


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
