import uuid

from fastapi import APIRouter, HTTPException, status

from app.core.deps import CurrentUser, DbSession, SettingsDep
from app.schemas.llm_provider import (
    LlmProviderCreateRequest,
    LlmProviderPublic,
    LlmProviderUpdateRequest,
)
from app.services.llm_provider_service import LlmProviderService

router = APIRouter(prefix="/llm-providers", tags=["llm-providers"])

_PROVIDER_VALIDATION_STATUS: dict[str, int] = {
    "invalid_api_key": status.HTTP_400_BAD_REQUEST,
    "invalid_base_url": status.HTTP_400_BAD_REQUEST,
    "invalid_base_url_scheme": status.HTTP_400_BAD_REQUEST,
    "invalid_base_url_host": status.HTTP_400_BAD_REQUEST,
    "invalid_base_url_credentials": status.HTTP_400_BAD_REQUEST,
}


def _raise_provider_validation(exc: ValueError) -> None:
    detail = str(exc) or "invalid_provider"
    code = _PROVIDER_VALIDATION_STATUS.get(detail, status.HTTP_400_BAD_REQUEST)
    raise HTTPException(status_code=code, detail=detail) from exc


def _to_public(svc: LlmProviderService, row) -> LlmProviderPublic:
    data = svc.to_public(row)
    return LlmProviderPublic(**data)


@router.get("", response_model=list[LlmProviderPublic])
async def list_providers(
    current_user: CurrentUser,
    db: DbSession,
    settings: SettingsDep,
) -> list[LlmProviderPublic]:
    svc = LlmProviderService(db, settings)
    rows = await svc.list_for_user(current_user.id)
    return [_to_public(svc, row) for row in rows]


@router.post("", response_model=LlmProviderPublic, status_code=status.HTTP_201_CREATED)
async def create_provider(
    body: LlmProviderCreateRequest,
    current_user: CurrentUser,
    db: DbSession,
    settings: SettingsDep,
) -> LlmProviderPublic:
    svc = LlmProviderService(db, settings)
    try:
        provider_id = uuid.UUID(body.id) if body.id else uuid.uuid4()
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="invalid_provider_id") from exc

    existing = await svc.get_for_user(provider_id, current_user.id)
    if existing is not None:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail="provider_id_exists")

    try:
        row = await svc.create(
            current_user.id,
            provider_id=provider_id,
            name=body.name,
            base_url=body.base_url,
            api_key=body.api_key,
            model=body.model,
            enabled=body.enabled,
            is_default=body.is_default,
        )
    except ValueError as exc:
        _raise_provider_validation(exc)
    await db.commit()
    return _to_public(svc, row)


def _parse_provider_id(provider_id: str) -> uuid.UUID:
    try:
        return uuid.UUID(provider_id)
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="invalid_provider_id") from exc


@router.patch("/{provider_id}", response_model=LlmProviderPublic)
async def update_provider(
    provider_id: str,
    body: LlmProviderUpdateRequest,
    current_user: CurrentUser,
    db: DbSession,
    settings: SettingsDep,
) -> LlmProviderPublic:
    svc = LlmProviderService(db, settings)
    pid = _parse_provider_id(provider_id)
    row = await svc.get_for_user(pid, current_user.id)
    if row is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="provider_not_found")

    try:
        row = await svc.update(
            row,
            name=body.name,
            base_url=body.base_url,
            api_key=body.api_key,
            model=body.model,
            enabled=body.enabled,
            is_default=body.is_default,
        )
    except ValueError as exc:
        _raise_provider_validation(exc)
    await db.commit()
    return _to_public(svc, row)


@router.delete("/{provider_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_provider(
    provider_id: str,
    current_user: CurrentUser,
    db: DbSession,
    settings: SettingsDep,
) -> None:
    svc = LlmProviderService(db, settings)
    pid = _parse_provider_id(provider_id)
    row = await svc.get_for_user(pid, current_user.id)
    if row is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="provider_not_found")
    await svc.delete(row)
    await db.commit()


@router.post("/{provider_id}/default", response_model=LlmProviderPublic)
async def set_default_provider(
    provider_id: str,
    current_user: CurrentUser,
    db: DbSession,
    settings: SettingsDep,
) -> LlmProviderPublic:
    svc = LlmProviderService(db, settings)
    pid = _parse_provider_id(provider_id)
    row = await svc.get_for_user(pid, current_user.id)
    if row is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="provider_not_found")
    row = await svc.set_default(row)
    await db.commit()
    return _to_public(svc, row)
