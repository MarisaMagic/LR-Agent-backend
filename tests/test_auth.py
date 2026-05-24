import pytest
from httpx import AsyncClient
from sqlalchemy.ext.asyncio import AsyncSession


@pytest.mark.asyncio
async def test_health_check(client: AsyncClient) -> None:
    response = await client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


@pytest.mark.asyncio
async def test_register_login_and_profile(
    client: AsyncClient,
    unique_email: str,
    test_password: str,
) -> None:
    register_response = await client.post(
        "/api/v1/auth/register",
        json={"email": unique_email, "password": test_password},
    )
    assert register_response.status_code == 201
    assert register_response.json()["message"] == "verification_email_sent"

    login_response = await client.post(
        "/api/v1/auth/login",
        json={"email": unique_email, "password": test_password},
    )
    assert login_response.status_code == 200
    payload = login_response.json()
    assert payload["token_type"] == "bearer"
    assert payload["user"]["email"] == unique_email
    assert payload["user"]["email_verified"] is False

    access_token = payload["access_token"]
    refresh_token = payload["refresh_token"]

    me_response = await client.get(
        "/api/v1/users/me",
        headers={"Authorization": f"Bearer {access_token}"},
    )
    assert me_response.status_code == 200
    assert me_response.json()["email"] == unique_email

    patch_response = await client.patch(
        "/api/v1/users/me",
        headers={"Authorization": f"Bearer {access_token}"},
        json={"username": "alice_test", "display_name": "Alice"},
    )
    assert patch_response.status_code == 200
    profile = patch_response.json()
    assert profile["username"] == "alice_test"
    assert profile["display_name"] == "Alice"

    refresh_response = await client.post(
        "/api/v1/auth/refresh",
        json={"refresh_token": refresh_token},
    )
    assert refresh_response.status_code == 200
    new_refresh_token = refresh_response.json()["refresh_token"]
    assert new_refresh_token != refresh_token

    logout_response = await client.post(
        "/api/v1/auth/logout",
        json={"refresh_token": new_refresh_token},
    )
    assert logout_response.status_code == 200

    stale_refresh = await client.post(
        "/api/v1/auth/refresh",
        json={"refresh_token": new_refresh_token},
    )
    assert stale_refresh.status_code == 401


@pytest.mark.asyncio
async def test_refresh_token_reuse_revokes_sessions(
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
    refresh_token = login_response.json()["refresh_token"]

    first_refresh = await client.post(
        "/api/v1/auth/refresh",
        json={"refresh_token": refresh_token},
    )
    assert first_refresh.status_code == 200
    new_refresh_token = first_refresh.json()["refresh_token"]

    reuse_response = await client.post(
        "/api/v1/auth/refresh",
        json={"refresh_token": refresh_token},
    )
    assert reuse_response.status_code == 401
    assert reuse_response.json()["detail"] == "token_reuse_detected"

    new_token_response = await client.post(
        "/api/v1/auth/refresh",
        json={"refresh_token": new_refresh_token},
    )
    assert new_token_response.status_code == 401


@pytest.mark.asyncio
async def test_revoke_all_sessions(
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
    refresh_token = login_response.json()["refresh_token"]

    revoke_response = await client.post(
        "/api/v1/auth/revoke-all-sessions",
        headers={"Authorization": f"Bearer {access_token}"},
    )
    assert revoke_response.status_code == 200

    refresh_after_revoke = await client.post(
        "/api/v1/auth/refresh",
        json={"refresh_token": refresh_token},
    )
    assert refresh_after_revoke.status_code == 401


@pytest.mark.asyncio
async def test_resend_verification_email(
    client: AsyncClient,
    db_session: AsyncSession,
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

    resend_response = await client.post(
        "/api/v1/auth/resend-verification-email",
        headers={"Authorization": f"Bearer {access_token}"},
    )
    assert resend_response.status_code == 200
    assert resend_response.json()["message"] == "verification_email_sent"

    from app.services.user_service import UserService

    user = await UserService(db_session).get_by_email(unique_email)
    assert user is not None
    user.email_verified = True

    already_verified = await client.post(
        "/api/v1/auth/resend-verification-email",
        headers={"Authorization": f"Bearer {access_token}"},
    )
    assert already_verified.status_code == 400
    assert already_verified.json()["detail"] == "email_already_verified"
