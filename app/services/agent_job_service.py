from redis.asyncio import Redis

from app.core.config import Settings
from app.schemas.agent import JobState


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

    def _state_key(self, user_id: str, client_job_id: str) -> str:
        return f"agent:job:{user_id}:{client_job_id}:state"

    async def register_job(self, user_id: str, client_job_id: str) -> None:
        ttl = self.settings.agent_job_cancel_ttl_seconds
        await self.redis.set(self._owner_key(user_id, client_job_id), user_id, ex=ttl)
        await self.redis.set(self._state_key(user_id, client_job_id), JobState.REGISTERED.value, ex=ttl)

    async def mark_cancelled(self, user_id: str, client_job_id: str) -> bool:
        owner = await self.redis.get(self._owner_key(user_id, client_job_id))
        if owner is None:
            return False
        ttl = self.settings.agent_job_cancel_ttl_seconds
        await self.redis.set(self._cancelled_key(user_id, client_job_id), "1", ex=ttl)
        await self.redis.set(self._state_key(user_id, client_job_id), JobState.CANCELLED.value, ex=ttl)
        return True

    async def is_cancelled(self, user_id: str, client_job_id: str) -> bool:
        ttl_key = self._cancelled_key(user_id, client_job_id)
        if await self.redis.get(ttl_key) is not None:
            return True
        legacy = await self.redis.get(self._legacy_cancelled_key(client_job_id))
        return legacy is not None

    async def set_state(self, user_id: str, client_job_id: str, state: JobState) -> None:
        """更新任务状态。"""
        ttl = self.settings.agent_job_cancel_ttl_seconds
        await self.redis.set(self._state_key(user_id, client_job_id), state.value, ex=ttl)

    async def get_state(self, user_id: str, client_job_id: str) -> JobState | None:
        """读取任务当前状态。"""
        val = await self.redis.get(self._state_key(user_id, client_job_id))
        return JobState(val) if val else None
