from app.schemas.agent import StreamEventPayload
from app.services.agent_chat_blocks import (
    apply_stream_event_to_blocks,
    blocks_to_preview,
    collapse_assistant_blocks,
    finalize_pipeline_steps_in_blocks,
)


def test_annotation_progress_creates_pipeline_and_upserts_stage() -> None:
    blocks: list[dict] = []
    event = StreamEventPayload(
        type="annotation_progress",
        stage="prepare",
        message="准备中",
        status="running",
    )
    blocks = apply_stream_event_to_blocks(blocks, event)
    assert len(blocks) == 1
    assert blocks[0]["type"] == "annotation_pipeline"
    assert blocks[0]["steps"][0]["stage"] == "prepare"

    done = StreamEventPayload(
        type="annotation_progress",
        stage="prepare",
        message="准备完成",
        status="done",
    )
    blocks = apply_stream_event_to_blocks(blocks, done)
    assert len(blocks[0]["steps"]) == 1
    assert blocks[0]["steps"][0]["status"] == "done"


def test_annotation_proposal_collapses_pipeline_and_sets_preview() -> None:
    blocks = apply_stream_event_to_blocks(
        [],
        StreamEventPayload(
            type="annotation_progress",
            stage="workers",
            message="处理中",
            status="running",
        ),
    )
    proposal = {
        "id": "p1",
        "summary": "已为 3 张图片生成标注",
        "stats": {"succeeded": 3, "totalBoxes": 12},
    }
    blocks = apply_stream_event_to_blocks(
        blocks,
        StreamEventPayload(type="annotation_proposal", proposal=proposal),
    )
    pipeline = next(b for b in blocks if b["type"] == "annotation_pipeline")
    assert pipeline["collapsed"] is True
    assert all(s["status"] != "running" for s in pipeline["steps"])
    assert any(b["type"] == "annotation_proposal" for b in blocks)
    assert "3 张" in blocks_to_preview(blocks)


def test_finalize_pipeline_steps_marks_running_as_done() -> None:
    blocks = collapse_assistant_blocks(
        [
            {
                "type": "annotation_pipeline",
                "collapsed": False,
                "steps": [
                    {"stage": "prepare", "status": "done", "message": "ok"},
                    {"stage": "workers", "status": "running", "message": "处理中"},
                ],
            },
        ],
    )
    finalized = finalize_pipeline_steps_in_blocks(blocks, terminal_status="done")
    pipeline = next(b for b in finalized if b["type"] == "annotation_pipeline")
    assert pipeline["collapsed"] is True
    assert all(s["status"] != "running" for s in pipeline["steps"])
    assert pipeline["steps"][1]["status"] == "done"


def test_worker_steps_upsert_by_image_path() -> None:
    blocks: list[dict] = []
    blocks = apply_stream_event_to_blocks(
        blocks,
        StreamEventPayload(
            type="annotation_progress",
            stage="worker",
            message="处理中 (1/2)：a/1.jpg",
            status="running",
            image_path="a/1.jpg",
        ),
    )
    blocks = apply_stream_event_to_blocks(
        blocks,
        StreamEventPayload(
            type="annotation_progress",
            stage="worker",
            message="完成：a/1.jpg",
            status="done",
            detail="映射 3 框",
            image_path="a/1.jpg",
        ),
    )
    blocks = apply_stream_event_to_blocks(
        blocks,
        StreamEventPayload(
            type="annotation_progress",
            stage="worker",
            message="完成：a/2.jpg",
            status="done",
            detail="映射 5 框",
            image_path="a/2.jpg",
        ),
    )
    pipeline = next(b for b in blocks if b["type"] == "annotation_pipeline")
    worker_steps = [s for s in pipeline["steps"] if s["stage"] == "worker"]
    assert len(worker_steps) == 2
    assert all(s["status"] == "done" for s in worker_steps)
    by_path = {s["imagePath"]: s for s in worker_steps}
    assert by_path["a/1.jpg"]["detail"] == "映射 3 框"
    assert by_path["a/2.jpg"]["detail"] == "映射 5 框"
