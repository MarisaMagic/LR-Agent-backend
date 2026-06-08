import uuid
from datetime import datetime, timezone

import pytest
from httpx import AsyncClient

from app.models.agent_message import AgentMessage
from app.services.agent_chat_repository import AgentChatRepository


@pytest.mark.asyncio
async def test_list_sessions_cursor_and_summary(
    client: AsyncClient,
    db_session,
    fake_redis,
    unique_email: str,
    test_password: str,
) -> None:
    from app.core.config import get_settings

    await client.post(
        "/api/v1/auth/register",
        json={"email": unique_email, "password": test_password},
    )
    login = await client.post(
        "/api/v1/auth/login",
        json={"email": unique_email, "password": test_password},
    )
    headers = {"Authorization": f"Bearer {login.json()['access_token']}"}
    me = await client.get("/api/v1/users/me", headers=headers)
    user_id = uuid.UUID(me.json()["id"])

    settings = get_settings()
    repo = AgentChatRepository(db_session, fake_redis, settings)

    now = datetime.now(timezone.utc)
    for index in range(3):
        session_id = f"session-page-{index}"
        await repo.create_session(
            user_id,
            session_id=session_id,
            title=f"对话 {index}",
        )
        # list_sessions 会过滤无消息的 session，每会话至少一条消息
        db_session.add(
            AgentMessage(
                id=f"msg-page-{index}",
                session_id=session_id,
                user_id=user_id,
                role="user",
                sort_index=0,
                blocks_json=[{"type": "text", "content": f"内容 {index}"}],
                status="done",
                created_at=now,
                updated_at=now,
            ),
        )
    await db_session.commit()

    first = await client.get("/api/v1/agent/sessions?limit=2", headers=headers)
    assert first.status_code == 200
    body = first.json()
    assert len(body["sessions"]) == 2
    assert body["has_more"] is True
    assert body["next_cursor"]
    assert body["sessions"][0]["message_count"] == 1
    assert body["sessions"][0]["message_ids"] == []

    second = await client.get(
        f"/api/v1/agent/sessions?limit=2&cursor={body['next_cursor']}",
        headers=headers,
    )
    assert second.status_code == 200
    second_body = second.json()
    assert len(second_body["sessions"]) == 1
    assert second_body["has_more"] is False


@pytest.mark.asyncio
async def test_session_messages_pagination(
    client: AsyncClient,
    db_session,
    fake_redis,
    unique_email: str,
    test_password: str,
) -> None:
    from app.core.config import get_settings
    from app.models.agent_message import AgentMessage
    from datetime import datetime, timezone

    await client.post(
        "/api/v1/auth/register",
        json={"email": unique_email, "password": test_password},
    )
    login = await client.post(
        "/api/v1/auth/login",
        json={"email": unique_email, "password": test_password},
    )
    headers = {"Authorization": f"Bearer {login.json()['access_token']}"}
    me = await client.get("/api/v1/users/me", headers=headers)
    user_id = uuid.UUID(me.json()["id"])

    settings = get_settings()
    repo = AgentChatRepository(db_session, fake_redis, settings)
    session_id = "session-msg-page"
    await repo.create_session(user_id, session_id=session_id, title="分页测试")
    now = datetime.now(timezone.utc)
    for index in range(5):
        db_session.add(
            AgentMessage(
                id=f"msg-{index}",
                session_id=session_id,
                user_id=user_id,
                role="user" if index % 2 == 0 else "assistant",
                sort_index=index,
                blocks_json=[{"type": "text", "content": f"内容 {index}"}],
                status="done",
                created_at=now,
                updated_at=now,
            ),
        )
    await db_session.commit()

    first = await client.get(
        f"/api/v1/agent/sessions/{session_id}?limit=2",
        headers=headers,
    )
    assert first.status_code == 200
    page = first.json()
    assert page["has_more_before"] is True
    assert len(page["messages"]) == 2
    assert page["session"]["message_count"] == 5
    assert page["messages"][0]["id"] == "msg-3"
    assert page["messages"][1]["id"] == "msg-4"

    oldest_id = page["messages"][0]["id"]
    older = await client.get(
        f"/api/v1/agent/sessions/{session_id}?limit=2&before_message_id={oldest_id}",
        headers=headers,
    )
    assert older.status_code == 200
    older_page = older.json()
    assert len(older_page["messages"]) == 2
    assert older_page["messages"][0]["id"] == "msg-1"
    assert older_page["messages"][1]["id"] == "msg-2"


@pytest.mark.asyncio
async def test_append_user_message_rejects_assistant_id_conflict(
    client: AsyncClient,
    db_session,
    fake_redis,
    unique_email: str,
    test_password: str,
) -> None:
    from datetime import datetime, timezone

    from app.core.config import get_settings
    from app.models.agent_message import AgentMessage

    await client.post(
        "/api/v1/auth/register",
        json={"email": unique_email, "password": test_password},
    )
    login = await client.post(
        "/api/v1/auth/login",
        json={"email": unique_email, "password": test_password},
    )
    me = await client.get(
        "/api/v1/users/me",
        headers={"Authorization": f"Bearer {login.json()['access_token']}"},
    )
    user_id = uuid.UUID(me.json()["id"])

    settings = get_settings()
    repo = AgentChatRepository(db_session, fake_redis, settings)
    session_id = "session-user-id-conflict"
    await repo.create_session(user_id, session_id=session_id, title="conflict-test")
    now = datetime.now(timezone.utc)
    db_session.add(
        AgentMessage(
            id="msg-assistant",
            session_id=session_id,
            user_id=user_id,
            role="assistant",
            sort_index=0,
            blocks_json=[],
            status="streaming",
            created_at=now,
            updated_at=now,
        ),
    )
    await db_session.flush()

    with pytest.raises(ValueError, match="user_message_id_conflict"):
        await repo._append_user_message(
            user_id=user_id,
            session_id=session_id,
            message_id="msg-assistant",
            content="第二轮用户问题",
            provider_id="prov",
            model="gpt",
        )


@pytest.mark.asyncio
async def test_decode_session_cursor_invalid() -> None:
    from app.services.agent_session_cursor import decode_session_cursor

    with pytest.raises(ValueError):
        decode_session_cursor("not-a-cursor")
