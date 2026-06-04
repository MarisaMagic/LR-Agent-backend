from uuid import UUID

from fastapi import HTTPException, status
from redis.asyncio import Redis

from app.core.config import Settings


async def _check_limit(
    redis: Redis,
    key: str,
    *,
    limit: int,
    window_seconds: int,
) -> None:
    current = await redis.incr(key)
    if current == 1:
        await redis.expire(key, window_seconds)
    if current > limit:
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail="rate_limit_exceeded",
        )


async def check_agent_stream_limit(redis: Redis, settings: Settings, user_id: UUID) -> None:
    uid = str(user_id)
    await _check_limit(
        redis,
        f"lr:ratelimit:agent:stream:min:{uid}",
        limit=settings.agent_stream_rate_limit_per_minute,
        window_seconds=60,
    )
    await _check_limit(
        redis,
        f"lr:ratelimit:agent:stream:day:{uid}",
        limit=settings.agent_stream_rate_limit_per_day,
        window_seconds=86_400,
    )


async def check_agent_session_write_limit(
    redis: Redis,
    settings: Settings,
    user_id: UUID,
) -> None:
    uid = str(user_id)
    limit = settings.agent_session_write_rate_limit_per_hour
    if settings.is_development:
        limit = max(limit, 500)
    await _check_limit(
        redis,
        f"lr:ratelimit:agent:session:hour:{uid}",
        limit=limit,
        window_seconds=3600,
    )
