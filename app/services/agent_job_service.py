from redis.asyncio import Redis

from app.core.config import Settings


class AgentJobService:
    def __init__(self, redis: Redis, settings: Settings) -> None:
        self.redis = redis
        self.settings = settings

    def _owner_key(self, user_id: str, client_job_id: str) -> str:
        return f"agent:job:{user_id}:{client_job_id}:owner"

    def _cancelled_key(self, user_id: str, client_job_id: str) -> str:
        return f"agent:job:{user_id}:{client_job_id}:cancelled"

    def _legacy_cancelled_key(self, client_job_id: str) -> str:
        return f"agent:job:{client_job_id}:cancelled"

    async def register_job(self, user_id: str, client_job_id: str) -> None:
        ttl = self.settings.agent_job_cancel_ttl_seconds
        await self.redis.set(self._owner_key(user_id, client_job_id), user_id, ex=ttl)

    async def mark_cancelled(self, user_id: str, client_job_id: str) -> bool:
        owner = await self.redis.get(self._owner_key(user_id, client_job_id))
        if owner is None:
            return False
        ttl = self.settings.agent_job_cancel_ttl_seconds
        await self.redis.set(self._cancelled_key(user_id, client_job_id), "1", ex=ttl)
        return True

    async def is_cancelled(self, user_id: str, client_job_id: str) -> bool:
        ttl_key = self._cancelled_key(user_id, client_job_id)
        if await self.redis.get(ttl_key) is not None:
            return True
        legacy = await self.redis.get(self._legacy_cancelled_key(client_job_id))
        return legacy is not None
