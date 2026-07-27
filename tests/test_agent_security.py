import pytest
from httpx import AsyncClient


@pytest.mark.asyncio
async def test_chat_cancel_is_public(client: AsyncClient) -> None:
    response = await client.post(
        "/api/v1/agent/chat/cancel",
        json={"client_job_id": "job-test-1"},
    )
    assert response.status_code == 200
    assert response.json()["ok"] is True
