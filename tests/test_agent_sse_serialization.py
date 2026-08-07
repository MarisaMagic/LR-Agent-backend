import json

from app.schemas.agent import StreamEventPayload


def test_file_proposal_sse_uses_frontend_field_names():
    event = StreamEventPayload(
        type="file_proposal",
        summary="Readme",
        content="# Hello",
        image_path="docs/readme.md",
    )
    data = event.to_sse_dict()
    assert data["type"] == "file_proposal"
    assert data["suggestedRelativePath"] == "docs/readme.md"
    assert data["title"] == "Readme"
    assert data["content"] == "# Hello"
    assert "image_path" not in data


def test_chat_stream_serializes_with_to_sse_dict_shape():
    event = StreamEventPayload(
        type="file_proposal_start",
        summary="notes/todo.md",
        image_path="notes/todo.md",
        detail="0",
    )
    payload = json.dumps(event.to_sse_dict(), ensure_ascii=False)
    parsed = json.loads(payload)
    assert parsed["suggestedRelativePath"] == "notes/todo.md"
    assert parsed["title"] == "notes/todo.md"
