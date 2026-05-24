from fastapi import HTTPException, Request, status
from redis.asyncio import Redis
from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint
from starlette.responses import Response

from app.core.config import Settings, get_settings


class RateLimitMiddleware(BaseHTTPMiddleware):
    """Simple Redis-backed rate limiter for auth endpoints."""

    RULES: dict[str, tuple[int, int]] = {
        "/api/v1/auth/register": (5, 3600),
        "/api/v1/auth/login": (10, 900),
        "/api/v1/auth/verify-email": (3, 3600),
        "/api/v1/auth/resend-verification-email": (3, 3600),
        "/api/v1/auth/forgot-password": (3, 3600),
        "/api/v1/auth/reset-password": (3, 3600),
        "/api/v1/users/me/delete-account": (3, 3600),
    }

    def __init__(self, app, redis: Redis, settings: Settings | None = None) -> None:
        super().__init__(app)
        self.redis = redis
        self.settings = settings or get_settings()

    async def dispatch(self, request: Request, call_next: RequestResponseEndpoint) -> Response:
        if request.method == "OPTIONS":
            return await call_next(request)

        rule = self.RULES.get(request.url.path)
        if rule is None:
            return await call_next(request)

        limit, window_seconds = rule
        client_ip = request.client.host if request.client else "unknown"
        key = f"lr:ratelimit:{request.url.path}:{client_ip}"

        current = await self.redis.incr(key)
        if current == 1:
            await self.redis.expire(key, window_seconds)

        if current > limit:
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail="rate_limit_exceeded",
            )

        return await call_next(request)
