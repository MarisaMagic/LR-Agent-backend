from app.schemas.agent import StreamEventPayload
from app.services.agent_chat_blocks import apply_stream_event_to_blocks, blocks_to_preview


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
