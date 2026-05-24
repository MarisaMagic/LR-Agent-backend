import io

import pytest
from httpx import AsyncClient
from PIL import Image


def _make_test_image(fmt: str = "PNG") -> bytes:
    image = Image.new("RGB", (512, 512), color=(120, 180, 240))
    buffer = io.BytesIO()
    image.save(buffer, format=fmt)
    return buffer.getvalue()


@pytest.mark.asyncio
async def test_avatar_upload_and_delete(
    client: AsyncClient,
    unique_email: str,
    test_password: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    stored: dict[str, str] = {}

    async def mock_upload(_self, user_id: str, file_bytes: bytes, content_type: str) -> tuple[str, str]:
        avatar_key = f"avatars/{user_id}/mock"
        avatar_url = f"http://test.local/{avatar_key}_512.webp"
        stored["key"] = avatar_key
        stored["url"] = avatar_url
        return avatar_url, avatar_key

    async def mock_delete(_self, avatar_key: str) -> None:
        stored.pop("key", None)

    from app.services import avatar_service

    monkeypatch.setattr(avatar_service.AvatarService, "upload", mock_upload)
    monkeypatch.setattr(avatar_service.AvatarService, "delete", mock_delete)

    await client.post(
        "/api/v1/auth/register",
        json={"email": unique_email, "password": test_password},
    )
    login_response = await client.post(
        "/api/v1/auth/login",
        json={"email": unique_email, "password": test_password},
    )
    access_token = login_response.json()["access_token"]

    upload_response = await client.post(
        "/api/v1/users/me/avatar",
        headers={"Authorization": f"Bearer {access_token}"},
        files={"file": ("avatar.png", _make_test_image(), "image/png")},
    )
    assert upload_response.status_code == 200
    profile = upload_response.json()
    assert profile["avatar_url"] == stored["url"]

    delete_response = await client.delete(
        "/api/v1/users/me/avatar",
        headers={"Authorization": f"Bearer {access_token}"},
    )
    assert delete_response.status_code == 200
    assert delete_response.json()["avatar_url"] is None


@pytest.mark.asyncio
async def test_delete_account_soft_deactivate(
    client: AsyncClient,
    unique_email: str,
    test_password: str,
) -> None:
    await client.post(
        "/api/v1/auth/register",
        json={"email": unique_email, "password": test_password},
    )
    login_response = await client.post(
        "/api/v1/auth/login",
        json={"email": unique_email, "password": test_password},
    )
    access_token = login_response.json()["access_token"]

    wrong_password = await client.post(
        "/api/v1/users/me/delete-account",
        headers={"Authorization": f"Bearer {access_token}"},
        json={"password": "WrongPass123!"},
    )
    assert wrong_password.status_code == 401

    delete_response = await client.post(
        "/api/v1/users/me/delete-account",
        headers={"Authorization": f"Bearer {access_token}"},
        json={"password": test_password},
    )
    assert delete_response.status_code == 200
    assert delete_response.json()["message"] == "account_deleted"

    login_after_delete = await client.post(
        "/api/v1/auth/login",
        json={"email": unique_email, "password": test_password},
    )
    assert login_after_delete.status_code == 401

    reregister_response = await client.post(
        "/api/v1/auth/register",
        json={"email": unique_email, "password": test_password},
    )
    assert reregister_response.status_code == 201
