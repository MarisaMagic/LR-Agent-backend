import pytest
from httpx import AsyncClient
from redis.asyncio import Redis

from app.core.config import get_settings
from app.services.agent_job_service import AgentJobService


@pytest.mark.asyncio
async def test_chat_cancel_requires_auth(client: AsyncClient) -> None:
    response = await client.post(
        "/api/v1/agent/chat/cancel",
        json={"client_job_id": "job-test-1"},
    )
    assert response.status_code == 401


@pytest.mark.asyncio
async def test_agent_job_cancel_ownership(fake_redis: Redis) -> None:
    settings = get_settings()
    jobs = AgentJobService(fake_redis, settings)
    await jobs.register_job("user-a", "job-1")

    assert await jobs.mark_cancelled("user-a", "job-1") is True
    assert await jobs.is_cancelled("user-a", "job-1") is True

    await jobs.register_job("user-a", "job-2")
    assert await jobs.mark_cancelled("user-b", "job-2") is False
    assert await jobs.is_cancelled("user-b", "job-2") is False
