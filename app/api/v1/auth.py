from fastapi import APIRouter, HTTPException, Request, status
from fastapi.responses import HTMLResponse

from app.core.deps import CurrentUser, DbSession, RedisClient, SettingsDep
from app.schemas.user import (
    ForgotPasswordRequest,
    MessageResponse,
    ResetPasswordRequest,
    TokenRefreshRequest,
    TokenResponse,
    UserLoginRequest,
    UserRegisterRequest,
    VerifyEmailRequest,
)
from app.services.auth_service import AuthService
from app.services.verify_html import render_verify_page

router = APIRouter(prefix="/auth", tags=["auth"])


def _client_ip(request: Request) -> str | None:
    if request.client is None:
        return None
    return request.client.host


def _device_info(request: Request) -> str | None:
    return request.headers.get("user-agent")


@router.post("/register", response_model=MessageResponse, status_code=status.HTTP_201_CREATED)
async def register(
    body: UserRegisterRequest,
    db: DbSession,
    redis: RedisClient,
    settings: SettingsDep,
) -> MessageResponse:
    service = AuthService(db, redis, settings)
    try:
        await service.register(body.email, body.password)
    except ValueError as exc:
        if str(exc) == "email_already_registered":
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail="email_already_registered",
            ) from exc
        raise
    return MessageResponse(message="verification_email_sent")


@router.post("/resend-verification-email", response_model=MessageResponse)
async def resend_verification_email(
    current_user: CurrentUser,
    db: DbSession,
    redis: RedisClient,
    settings: SettingsDep,
) -> MessageResponse:
    service = AuthService(db, redis, settings)
    try:
        await service.resend_verification_email(current_user.id)
    except ValueError as exc:
        detail = str(exc)
        if detail == "email_already_verified":
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="email_already_verified",
            ) from exc
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="invalid_credentials",
        ) from exc
    return MessageResponse(message="verification_email_sent")


@router.post("/verify-email", response_model=MessageResponse)
async def verify_email(
    body: VerifyEmailRequest,
    db: DbSession,
    redis: RedisClient,
    settings: SettingsDep,
) -> MessageResponse:
    service = AuthService(db, redis, settings)
    try:
        await service.verify_email(body.token)
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="invalid_or_expired_token",
        ) from exc
    return MessageResponse(message="email_verified")


@router.get("/verify-email", response_class=HTMLResponse)
async def verify_email_page(
    token: str,
    db: DbSession,
    redis: RedisClient,
    settings: SettingsDep,
) -> HTMLResponse:
    """Browser-friendly verification endpoint (mobile/desktop email links)."""
    service = AuthService(db, redis, settings)
    try:
        await service.verify_email(token)
    except ValueError:
        return HTMLResponse(
            render_verify_page(
                "验证失败",
                "链接无效或已过期，请重新注册或联系支持。",
                success=False,
            ),
            status_code=status.HTTP_400_BAD_REQUEST,
        )
    return HTMLResponse(
        render_verify_page(
            "验证成功",
            "你的邮箱已成功验证。",
            success=True,
        )
    )


@router.post("/login", response_model=TokenResponse)
async def login(
    body: UserLoginRequest,
    request: Request,
    db: DbSession,
    redis: RedisClient,
    settings: SettingsDep,
) -> TokenResponse:
    service = AuthService(db, redis, settings)
    try:
        result = await service.login(
            body.email,
            body.password,
            ip_address=_client_ip(request),
            device_info=_device_info(request),
        )
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="invalid_credentials",
        ) from exc
    return TokenResponse(**result)


@router.post("/refresh", response_model=TokenResponse)
async def refresh_token(
    body: TokenRefreshRequest,
    db: DbSession,
    redis: RedisClient,
    settings: SettingsDep,
) -> TokenResponse:
    service = AuthService(db, redis, settings)
    try:
        result = await service.refresh(body.refresh_token)
    except ValueError as exc:
        detail = str(exc)
        if detail not in {"invalid_token", "invalid_credentials", "token_reuse_detected"}:
            detail = "invalid_token"
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=detail,
        ) from exc
    return TokenResponse(**result)


@router.post("/logout", response_model=MessageResponse)
async def logout(
    body: TokenRefreshRequest,
    db: DbSession,
    redis: RedisClient,
    settings: SettingsDep,
) -> MessageResponse:
    service = AuthService(db, redis, settings)
    await service.logout(body.refresh_token)
    return MessageResponse(message="logged_out")


@router.post("/revoke-all-sessions", response_model=MessageResponse)
async def revoke_all_sessions(
    current_user: CurrentUser,
    db: DbSession,
    redis: RedisClient,
    settings: SettingsDep,
) -> MessageResponse:
    service = AuthService(db, redis, settings)
    await service.revoke_all_sessions(str(current_user.id))
    return MessageResponse(message="all_sessions_revoked")


@router.post("/forgot-password", response_model=MessageResponse)
async def forgot_password(
    body: ForgotPasswordRequest,
    db: DbSession,
    redis: RedisClient,
    settings: SettingsDep,
) -> MessageResponse:
    service = AuthService(db, redis, settings)
    await service.forgot_password(body.email)
    return MessageResponse(message="reset_email_sent_if_account_exists")


@router.post("/reset-password", response_model=MessageResponse)
async def reset_password(
    body: ResetPasswordRequest,
    db: DbSession,
    redis: RedisClient,
    settings: SettingsDep,
) -> MessageResponse:
    service = AuthService(db, redis, settings)
    try:
        await service.reset_password(body.token, body.password)
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="invalid_or_expired_token",
        ) from exc
    return MessageResponse(message="password_reset_success")
