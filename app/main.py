from contextlib import asynccontextmanager
import logging

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from redis.asyncio import Redis

from app.api.v1.router import api_router
from app.core.config import get_settings
from app.db.session import get_redis_pool
from app.middleware.rate_limit import RateLimitMiddleware
from app.services.avatar_service import AvatarService

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    if settings.app_env != "testing":
        try:
            await AvatarService(settings).ensure_bucket()
            logger.info("MinIO bucket ready: %s", settings.minio_bucket)
        except Exception as exc:
            logger.warning("MinIO bucket initialization skipped: %s", exc)

    yield

    pool = get_redis_pool()
    await pool.disconnect()


def create_app() -> FastAPI:
    settings = get_settings()

    app = FastAPI(
        title=settings.app_name,
        debug=settings.debug,
        lifespan=lifespan,
    )

    cors_kwargs: dict = {
        "allow_origins": settings.cors_origins,
        "allow_credentials": True,
        "allow_methods": ["*"],
        "allow_headers": ["*"],
    }
    if settings.is_development:
        cors_kwargs["allow_origin_regex"] = (
            r"http://(localhost|127\.0\.0\.1|192\.168\.\d{1,3}\.\d{1,3}|10\.\d{1,3}\.\d{1,3}\.\d{1,3}):\d+"
        )

    app.add_middleware(CORSMiddleware, **cors_kwargs)

    redis = Redis(connection_pool=get_redis_pool())
    if settings.app_env != "testing":
        app.add_middleware(RateLimitMiddleware, redis=redis)

    app.include_router(api_router, prefix=settings.api_v1_prefix)

    @app.get("/health", tags=["health"])
    async def health_check() -> dict[str, str]:
        return {"status": "ok"}

    return app


app = create_app()
