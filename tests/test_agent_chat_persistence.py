import uuid
from datetime import datetime, timezone

import pytest
from sqlalchemy import select

from app.core.config import get_settings
from app.models.agent_message import AgentMessage
from app.models.user import User
from app.schemas.agent import AnnotationRunEventsRequest, StreamEventPayload
from app.services.agent_chat_repository import AgentChatRepository
from app.services.annotation_run_service import AnnotationRunService


async def _seed_user(db_session, email_suffix: str) -> uuid.UUID:
    user_id = uuid.uuid4()
    db_session.add(
        User(
            id=user_id,
            email=f"persist-{email_suffix}@example.com",
            email_verified=True,
            password_hash="hash",
            is_active=True,
        ),
    )
    await db_session.flush()
    return user_id


async def _seed_assistant_message(
    db_session,
    *,
    user_id: uuid.UUID,
    session_id: str,
    message_id: str,
    blocks_json: list | None = None,
) -> None:
    now = datetime.now(timezone.utc)
    db_session.add(
        AgentMessage(
            id=message_id,
            session_id=session_id,
            user_id=user_id,
            role="assistant",
            sort_index=1,
            blocks_json=blocks_json or [],
            status="streaming",
            created_at=now,
            updated_at=now,
        ),
    )
    await db_session.flush()


@pytest.mark.asyncio
async def test_file_proposal_pg_sync_when_cache_hit(db_session, fake_redis) -> None:
    settings = get_settings()
    repo = AgentChatRepository(db_session, fake_redis, settings)
    user_id = await _seed_user(db_session, "file-proposal")
    session_id = "sess-file-proposal"
    message_id = "msg-file-proposal"

    await repo.create_session(user_id, session_id=session_id, title="持久化测试")
    await _seed_assistant_message(
        db_session,
        user_id=user_id,
        session_id=session_id,
        message_id=message_id,
    )
    await repo.cache.set_message_ids(str(user_id), session_id, [message_id])
    await repo.cache.set_message(
        str(user_id),
        session_id,
        {
            "id": message_id,
            "session_id": session_id,
            "role": "assistant",
            "blocks": [],
            "status": "streaming",
            "provider_id": "",
            "model": "",
            "error": None,
            "created_at_ms": 1,
            "updated_at_ms": 1,
            "sort_index": 1,
        },
    )

    await repo.apply_stream_event(
        user_id,
        session_id,
        message_id,
        StreamEventPayload(
            type="file_proposal",
            summary="数据报告",
            content="# 报告",
            image_path="reports/summary.md",
        ),
    )

    result = await db_session.execute(
        select(AgentMessage).where(AgentMessage.id == message_id),
    )
    row = result.scalar_one()
    assert any(block.get("type") == "file_proposal" for block in row.blocks_json)

    cached = await repo.cache.get_message(str(user_id), message_id)
    assert cached is not None
    assert any(block.get("type") == "file_proposal" for block in cached.get("blocks", []))


@pytest.mark.asyncio
async def test_text_delta_cache_hit_does_not_sync_pg(db_session, fake_redis) -> None:
    settings = get_settings()
    repo = AgentChatRepository(db_session, fake_redis, settings)
    user_id = await _seed_user(db_session, "text-delta")
    session_id = "sess-text-delta"
    message_id = "msg-text-delta"

    await repo.create_session(user_id, session_id=session_id, title="文本增量")
    await _seed_assistant_message(
        db_session,
        user_id=user_id,
        session_id=session_id,
        message_id=message_id,
    )
    await repo.cache.set_message_ids(str(user_id), session_id, [message_id])
    await repo.cache.set_message(
        str(user_id),
        session_id,
        {
            "id": message_id,
            "session_id": session_id,
            "role": "assistant",
            "blocks": [],
            "status": "streaming",
            "provider_id": "",
            "model": "",
            "error": None,
            "created_at_ms": 1,
            "updated_at_ms": 1,
            "sort_index": 1,
        },
    )

    await repo.apply_stream_event(
        user_id,
        session_id,
        message_id,
        StreamEventPayload(type="text_delta", content="hello"),
    )

    result = await db_session.execute(
        select(AgentMessage).where(AgentMessage.id == message_id),
    )
    row = result.scalar_one()
    assert row.blocks_json == []

    cached = await repo.cache.get_message(str(user_id), message_id)
    assert cached is not None
    assert cached["blocks"] == [{"type": "text", "content": "hello"}]


@pytest.mark.asyncio
async def test_annotation_run_ingest_events_persists_proposal(db_session, fake_redis) -> None:
    settings = get_settings()
    repo = AgentChatRepository(db_session, fake_redis, settings)
    service = AnnotationRunService(db_session, fake_redis, settings)
    user_id = await _seed_user(db_session, "ann-run")
    session_id = "sess-ann-run"
    message_id = "msg-ann-run"

    await repo.create_session(user_id, session_id=session_id, title="标注上报")
    await _seed_assistant_message(
        db_session,
        user_id=user_id,
        session_id=session_id,
        message_id=message_id,
    )

    class _User:
        id = user_id

    await service.ingest_events(
        _User(),  # type: ignore[arg-type]
        AnnotationRunEventsRequest(
            session_id=session_id,
            assistant_message_id=message_id,
            client_job_id="job-ann-run-1",
            seq=1,
            events=[
                {
                    "type": "annotation_proposal",
                    "proposal": {
                        "id": "p1",
                        "summary": "批量标注完成",
                        "stats": {"succeeded": 2, "totalBoxes": 5},
                    },
                },
            ],
        ),
    )

    result = await db_session.execute(
        select(AgentMessage).where(AgentMessage.id == message_id),
    )
    row = result.scalar_one()
    assert any(block.get("type") == "annotation_proposal" for block in row.blocks_json)

    cached = await repo.cache.get_message(str(user_id), message_id)
    assert cached is not None
    assert any(block.get("type") == "annotation_proposal" for block in cached.get("blocks", []))
