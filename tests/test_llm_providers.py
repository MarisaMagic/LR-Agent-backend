import uuid

import pytest
from httpx import AsyncClient


@pytest.mark.asyncio
async def test_llm_provider_crud(
    client: AsyncClient,
    unique_email: str,
    test_password: str,
) -> None:
    await client.post(
        "/api/v1/auth/register",
        json={"email": unique_email, "password": test_password},
    )
    login = await client.post(
        "/api/v1/auth/login",
        json={"email": unique_email, "password": test_password},
    )
    token = login.json()["access_token"]
    headers = {"Authorization": f"Bearer {token}"}

    provider_id = str(uuid.uuid4())
    create = await client.post(
        "/api/v1/llm-providers",
        headers=headers,
        json={
            "id": provider_id,
            "name": "Test",
            "base_url": "https://example.com/v1",
            "api_key": "sk-test-key-12345678",
            "model": "gpt-4o-mini",
            "enabled": True,
            "is_default": True,
        },
    )
    assert create.status_code == 201
    created = create.json()
    assert created["id"] == provider_id
    assert created["is_default"] is True
    assert "****" in created["api_key"]

    listing = await client.get("/api/v1/llm-providers", headers=headers)
    assert listing.status_code == 200
    assert len(listing.json()) == 1

    patch = await client.patch(
        f"/api/v1/llm-providers/{provider_id}",
        headers=headers,
        json={"name": "Renamed"},
    )
    assert patch.status_code == 200
    assert patch.json()["name"] == "Renamed"

    delete = await client.delete(
        f"/api/v1/llm-providers/{provider_id}",
        headers=headers,
    )
    assert delete.status_code == 204


@pytest.mark.asyncio
async def test_llm_provider_rejects_private_base_url(
    client: AsyncClient,
    unique_email: str,
    test_password: str,
) -> None:
    await client.post(
        "/api/v1/auth/register",
        json={"email": unique_email, "password": test_password},
    )
    login = await client.post(
        "/api/v1/auth/login",
        json={"email": unique_email, "password": test_password},
    )
    headers = {"Authorization": f"Bearer {login.json()['access_token']}"}

    blocked = await client.post(
        "/api/v1/llm-providers",
        headers=headers,
        json={
            "name": "Blocked",
            "base_url": "https://127.0.0.1/v1",
            "api_key": "sk-test-key-12345678",
            "model": "gpt-4o-mini",
            "enabled": True,
            "is_default": False,
        },
    )
    assert blocked.status_code == 400
    assert blocked.json()["detail"] == "invalid_base_url_host"


@pytest.mark.asyncio
async def test_llm_provider_patch_ignores_masked_api_key(
    client: AsyncClient,
    unique_email: str,
    test_password: str,
) -> None:
    await client.post(
        "/api/v1/auth/register",
        json={"email": unique_email, "password": test_password},
    )
    login = await client.post(
        "/api/v1/auth/login",
        json={"email": unique_email, "password": test_password},
    )
    headers = {"Authorization": f"Bearer {login.json()['access_token']}"}

    provider_id = str(uuid.uuid4())
    await client.post(
        "/api/v1/llm-providers",
        headers=headers,
        json={
            "id": provider_id,
            "name": "Test",
            "base_url": "https://example.com/v1",
            "api_key": "sk-original-key-123456",
            "model": "gpt-4o-mini",
            "enabled": True,
            "is_default": True,
        },
    )

    patch = await client.patch(
        f"/api/v1/llm-providers/{provider_id}",
        headers=headers,
        json={"name": "Renamed", "api_key": "sk-****3456"},
    )
    assert patch.status_code == 200
    assert patch.json()["name"] == "Renamed"
    assert "****" in patch.json()["api_key"]
