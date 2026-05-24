from fastapi import APIRouter, HTTPException, UploadFile, status

from app.core.deps import CurrentUser, DbSession, RedisClient, SettingsDep
from app.schemas.user import DeleteAccountRequest, MessageResponse, UserPublic, UserUpdateRequest
from app.services.auth_service import AuthService
from app.services.avatar_service import AvatarService
from app.services.user_service import UserService

router = APIRouter(prefix="/users", tags=["users"])

AVATAR_ERROR_STATUS = {
    "invalid_image_type": status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
    "invalid_image": status.HTTP_400_BAD_REQUEST,
    "file_too_large": status.HTTP_413_CONTENT_TOO_LARGE,
}


@router.get("/me", response_model=UserPublic)
async def get_me(current_user: CurrentUser) -> UserPublic:
    return UserPublic.model_validate(current_user)


@router.patch("/me", response_model=UserPublic)
async def update_me(
    body: UserUpdateRequest,
    current_user: CurrentUser,
    db: DbSession,
) -> UserPublic:
    service = UserService(db)

    if body.username is not None:
        existing = await service.get_by_username(body.username)
        if existing is not None and existing.id != current_user.id:
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail="username_already_taken",
            )

    user = await service.update_profile(
        current_user,
        username=body.username,
        display_name=body.display_name,
    )
    return UserPublic.model_validate(user)


@router.post("/me/avatar", response_model=UserPublic)
async def upload_avatar(
    file: UploadFile,
    current_user: CurrentUser,
    db: DbSession,
    settings: SettingsDep,
) -> UserPublic:
    content = await file.read()
    avatar_service = AvatarService(settings)
    user_service = UserService(db)

    try:
        avatar_url, avatar_key = await avatar_service.upload(
            str(current_user.id),
            content,
            file.content_type or "",
        )
    except ValueError as exc:
        detail = str(exc)
        raise HTTPException(
            status_code=AVATAR_ERROR_STATUS.get(detail, status.HTTP_400_BAD_REQUEST),
            detail=detail,
        ) from exc

    if current_user.avatar_key:
        await avatar_service.delete(current_user.avatar_key)

    user = await user_service.update_avatar(
        current_user,
        avatar_url=avatar_url,
        avatar_key=avatar_key,
    )
    return UserPublic.model_validate(user)


@router.delete("/me/avatar", response_model=UserPublic)
async def delete_avatar(
    current_user: CurrentUser,
    db: DbSession,
    settings: SettingsDep,
) -> UserPublic:
    if not current_user.avatar_key:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="avatar_not_found",
        )

    avatar_service = AvatarService(settings)
    await avatar_service.delete(current_user.avatar_key)

    user = await UserService(db).clear_avatar(current_user)
    return UserPublic.model_validate(user)


@router.post("/me/delete-account", response_model=MessageResponse)
async def delete_account(
    body: DeleteAccountRequest,
    current_user: CurrentUser,
    db: DbSession,
    redis: RedisClient,
    settings: SettingsDep,
) -> MessageResponse:
    if current_user.avatar_key:
        await AvatarService(settings).delete(current_user.avatar_key)

    service = AuthService(db, redis, settings)
    try:
        await service.delete_account(current_user.id, body.password)
    except ValueError as exc:
        if str(exc) == "invalid_credentials":
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="invalid_credentials",
            ) from exc
        raise
    return MessageResponse(message="account_deleted")
