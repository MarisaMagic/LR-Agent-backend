from typing import Any

from app.schemas.agent import StreamEventPayload

PERSISTABLE_STREAM_EVENT_TYPES = frozenset({
    "text_delta",
    "reasoning_delta",
    "tool_start",
    "tool_result",
    "annotation_progress",
    "annotation_proposal",
    "analysis_script_proposal",
    "file_proposal_start",
    "file_proposal_delta",
    "file_proposal",
    "document_proposal",
})

PG_SYNC_STREAM_EVENT_TYPES = frozenset({
    "annotation_proposal",
    "analysis_script_proposal",
    "file_proposal",
    "document_proposal",
})

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


ANALYSIS_PIPELINE_STAGE_LABELS: dict[str, str] = {
    "collect": "收集数据",
    "prepare": "生成脚本",
    "execute": "运行脚本",
    "summarize": "解读结果",
}

MUTATION_PIPELINE_STAGE_LABELS: dict[str, str] = {
    "prepare": "解析意图",
    "resolve": "定位目标",
}

REPORT_PIPELINE_STAGE_LABELS: dict[str, str] = {
    "collect": "收集数据",
    "prepare": "生成报告",
}

PIPELINE_STAGE_LABELS: dict[str, str] = {
    "prepare": "准备",
    "resolve": "定位目标",
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


def _label_for_pipeline_stage(stage: str, pipeline_kind: str = "batch") -> str:
    if pipeline_kind == "analysis":
        return ANALYSIS_PIPELINE_STAGE_LABELS.get(stage, stage)
    if pipeline_kind == "mutation":
        return MUTATION_PIPELINE_STAGE_LABELS.get(stage, stage)
    if pipeline_kind == "report":
        return REPORT_PIPELINE_STAGE_LABELS.get(stage, stage)
    return PIPELINE_STAGE_LABELS.get(stage, stage)


def _apply_annotation_progress_to_blocks(
    blocks: list[dict[str, Any]],
    event: StreamEventPayload,
) -> list[dict[str, Any]]:
    next_blocks = [dict(block) for block in blocks]
    pipeline_kind = (event.domain or "batch").strip() or "batch"
    incoming = {
        "stage": event.stage or "",
        "label": _label_for_pipeline_stage(event.stage or "", pipeline_kind),
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

    pipeline_idx = next(
        (
            i
            for i, b in enumerate(next_blocks)
            if b.get("type") == "annotation_pipeline"
            and (b.get("pipelineKind") or "batch") == pipeline_kind
        ),
        -1,
    )
    if pipeline_idx < 0:
        next_blocks.append(
            {
                "type": "annotation_pipeline",
                "collapsed": False,
                "steps": [incoming],
                "pipelineKind": pipeline_kind,
            },
        )
        return next_blocks

    block = next_blocks[pipeline_idx]
    next_blocks[pipeline_idx] = {
        **block,
        "pipelineKind": pipeline_kind,
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

    if event.type == "analysis_script_proposal":
        status = event.status or "pending"
        if status in ("done", "error"):
            for i, block in enumerate(next_blocks):
                if (
                    block.get("type") == "annotation_pipeline"
                    and (block.get("pipelineKind") or "batch") == "analysis"
                ):
                    next_blocks[i] = {
                        **block,
                        "collapsed": True,
                        "steps": [
                            {
                                **step,
                                "status": "done"
                                if step.get("status") == "running"
                                else step.get("status"),
                            }
                            for step in (block.get("steps") or [])
                        ],
                    }
        script_block = {
            "type": "analysis_script_proposal",
            "script": event.content or "",
            "explanation": event.detail or "",
            "status": status,
        }
        if event.result:
            script_block["result"] = event.result
        if event.message:
            script_block["error"] = event.message
        next_blocks = [b for b in next_blocks if b.get("type") != "analysis_script_proposal"]
        next_blocks.append(script_block)
        return next_blocks

    if event.type == "file_proposal" or event.type == "document_proposal":
        for i, block in enumerate(next_blocks):
            if (
                block.get("type") == "annotation_pipeline"
                and (block.get("pipelineKind") or "batch") == "report"
            ):
                next_blocks[i] = {
                    **block,
                    "collapsed": True,
                    "steps": [
                        {
                            **step,
                            "status": "done"
                            if step.get("status") == "running"
                            else step.get("status"),
                        }
                        for step in (block.get("steps") or [])
                    ],
                }
        doc_block = {
            "type": "file_proposal",
            "title": event.summary or "文件",
            "content": event.content or "",
            "suggestedRelativePath": event.image_path or "docs/document.md",
            "status": event.status or "pending",
        }
        # 按 suggestedRelativePath 去重更新，而非全量替换——支持同一消息中多个文件提案
        path = doc_block["suggestedRelativePath"]
        existing_idx = next(
            (
                i
                for i, b in enumerate(next_blocks)
                if b.get("type") in ("file_proposal", "document_proposal")
                and b.get("suggestedRelativePath") == path
            ),
            -1,
        )
        if existing_idx >= 0:
            existing_status = next_blocks[existing_idx].get("status")
            if existing_status and existing_status != "pending":
                doc_block["status"] = existing_status
            next_blocks[existing_idx] = doc_block
        else:
            next_blocks.append(doc_block)
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


def block_to_transcript_line(block: dict[str, Any]) -> str | None:
    """将非 text 提案块转为单行 transcript 摘要，供历史窗口复用。"""
    block_type = block.get("type")
    if block_type == "annotation_proposal":
        proposal = block.get("proposal") if isinstance(block.get("proposal"), dict) else {}
        stats = proposal.get("stats") if isinstance(proposal.get("stats"), dict) else {}
        succeeded = int(stats.get("succeeded") or 0)
        total_boxes = int(stats.get("totalBoxes") or stats.get("total_boxes") or 0)
        summary = str(proposal.get("summary") or "").strip()
        if summary:
            return summary
        if succeeded:
            hint = f"批量标注 {succeeded} 张"
            if total_boxes:
                hint += f" · {total_boxes} 框"
            return hint
        return "标注变更提案"
    if block_type == "analysis_script_proposal":
        explanation = str(block.get("explanation") or "").strip()
        if explanation:
            return f"[数据分析脚本] {explanation}"
        script = str(block.get("script") or "").strip().splitlines()
        first = script[0][:80] if script else ""
        return f"[数据分析脚本] {first}" if first else "[数据分析脚本]"
    if block_type in ("file_proposal", "document_proposal"):
        title = str(block.get("title") or "文件").strip()
        content = str(block.get("content") or "").strip()
        snippet = " ".join(content.split())[:120]
        if snippet:
            return f"[文件] {title}: {snippet}"
        return f"[文件] {title}"
    return None


def _truncate_preview_line(line: str, *, max_len: int) -> str:
    compact = " ".join(line.strip().split())
    return f"{compact[:max_len]}…" if len(compact) > max_len else compact


def blocks_to_text(blocks: list[dict[str, Any]]) -> str:
    parts: list[str] = []
    for block in blocks:
        if block.get("type") == "text":
            content = str(block.get("content", "")).strip()
            if content:
                parts.append(content)
    if parts:
        return "\n".join(parts)
    for block in blocks:
        line = block_to_transcript_line(block)
        if line:
            parts.append(line)
    return "\n".join(parts)


def blocks_to_preview(blocks: list[dict[str, Any]], *, max_len: int = 128) -> str:
    text = blocks_to_text(blocks)
    if text.strip():
        return _truncate_preview_line(text, max_len=max_len)

    for block in blocks:
        line = block_to_transcript_line(block)
        if line:
            return _truncate_preview_line(line, max_len=max_len)
    return ""
