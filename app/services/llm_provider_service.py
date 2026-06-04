import uuid
from datetime import datetime, timezone

from sqlalchemy import select, update
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import Settings
from app.models.llm_provider import LlmProvider
from app.services.crypto_service import mask_api_key
from app.services.llm_secrets import KEY_ID_ACTIVE, decrypt_api_key, encrypt_api_key, is_masked_api_key
from app.services.llm_vision_probe import probe_vision_capability
from app.services.url_safety import validate_llm_base_url


class LlmProviderService:
    def __init__(self, db: AsyncSession, settings: Settings) -> None:
        self.db = db
        self.settings = settings

    async def list_for_user(self, user_id: uuid.UUID) -> list[LlmProvider]:
        result = await self.db.execute(
            select(LlmProvider)
            .where(LlmProvider.user_id == user_id)
            .order_by(LlmProvider.updated_at.desc()),
        )
        return list(result.scalars().all())

    async def get_for_user(self, provider_id: uuid.UUID, user_id: uuid.UUID) -> LlmProvider | None:
        result = await self.db.execute(
            select(LlmProvider).where(
                LlmProvider.id == provider_id,
                LlmProvider.user_id == user_id,
            ),
        )
        return result.scalar_one_or_none()

    async def create(
        self,
        user_id: uuid.UUID,
        *,
        provider_id: uuid.UUID,
        name: str,
        base_url: str,
        api_key: str,
        model: str,
        enabled: bool,
        is_default: bool,
    ) -> LlmProvider:
        if is_masked_api_key(api_key):
            raise ValueError("invalid_api_key")

        normalized_url = validate_llm_base_url(base_url, self.settings)
        if is_default:
            await self._clear_default(user_id)

        row = LlmProvider(
            id=provider_id,
            user_id=user_id,
            name=name,
            base_url=normalized_url,
            api_key_encrypted=encrypt_api_key(self.settings, api_key),
            encryption_key_id=KEY_ID_ACTIVE,
            model=model,
            enabled=enabled,
            is_default=is_default,
        )
        self.db.add(row)
        await self.db.flush()
        await self.run_vision_probe(row)
        return row

    async def update(
        self,
        row: LlmProvider,
        *,
        name: str | None = None,
        base_url: str | None = None,
        api_key: str | None = None,
        model: str | None = None,
        enabled: bool | None = None,
        is_default: bool | None = None,
    ) -> LlmProvider:
        if name is not None:
            row.name = name
        if base_url is not None:
            row.base_url = validate_llm_base_url(base_url, self.settings)
        if api_key is not None and not is_masked_api_key(api_key):
            row.api_key_encrypted = encrypt_api_key(self.settings, api_key)
            row.encryption_key_id = KEY_ID_ACTIVE
        if model is not None:
            row.model = model
        if enabled is not None:
            row.enabled = enabled
        if is_default is not None:
            if is_default:
                await self._clear_default(row.user_id)
            row.is_default = is_default
        row.updated_at = datetime.now(timezone.utc)
        await self.db.flush()
        if base_url is not None or api_key is not None or model is not None:
            await self.run_vision_probe(row)
        return row

    async def run_vision_probe(self, row: LlmProvider) -> bool:
        """Call API with a tiny image; persist supports_vision on the provider row."""
        api_key = self.decrypt_api_key(row)
        supports, detail = await probe_vision_capability(row, api_key)
        row.supports_vision = supports
        row.vision_probe_detail = detail[:512]
        row.vision_probed_at = datetime.now(timezone.utc)
        row.updated_at = datetime.now(timezone.utc)
        await self.db.flush()
        return supports

    async def ensure_vision_probed(self, row: LlmProvider) -> bool:
        """Probe once if never probed (e.g. rows created before migration)."""
        if row.vision_probed_at is not None:
            return bool(row.supports_vision)
        return await self.run_vision_probe(row)

    async def delete(self, row: LlmProvider) -> None:
        await self.db.delete(row)
        await self.db.flush()

    async def set_default(self, row: LlmProvider) -> LlmProvider:
        await self._clear_default(row.user_id)
        row.is_default = True
        row.updated_at = datetime.now(timezone.utc)
        await self.db.flush()
        return row

    async def _clear_default(self, user_id: uuid.UUID) -> None:
        await self.db.execute(
            update(LlmProvider)
            .where(LlmProvider.user_id == user_id, LlmProvider.is_default.is_(True))
            .values(is_default=False),
        )

    def decrypt_api_key(self, row: LlmProvider) -> str:
        return decrypt_api_key(
            self.settings,
            row.api_key_encrypted,
            row.encryption_key_id,
        )

    def validate_provider_base_url(self, row: LlmProvider) -> None:
        validate_llm_base_url(row.base_url, self.settings)

    @staticmethod
    def to_public(row: LlmProvider, *, api_key_plain: str | None = None) -> dict:
        masked = mask_api_key(api_key_plain) if api_key_plain else "****"
        created_ms = int(row.created_at.timestamp() * 1000)
        updated_ms = int(row.updated_at.timestamp() * 1000)
        return {
            "id": str(row.id),
            "name": row.name,
            "base_url": row.base_url,
            "api_key": masked,
            "model": row.model,
            "enabled": row.enabled,
            "is_default": row.is_default,
            "supports_vision": bool(row.supports_vision),
            "vision_probed_at": (
                int(row.vision_probed_at.timestamp() * 1000) if row.vision_probed_at else None
            ),
            "vision_probe_detail": row.vision_probe_detail or "",
            "created_at": created_ms,
            "updated_at": updated_ms,
        }
