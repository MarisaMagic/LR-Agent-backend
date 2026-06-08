import os
from collections.abc import AsyncGenerator

import fakeredis.aioredis
import pytest
from httpx import ASGITransport, AsyncClient
from redis.asyncio import Redis
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.pool import NullPool

os.environ.setdefault("APP_ENV", "testing")
# 测试不走真实 SMTP（.env 中 SMTP_HOST 会被覆盖为空，EmailService 仅打 mock 日志）
os.environ["SMTP_HOST"] = ""

from cryptography.fernet import Fernet

os.environ.setdefault("LLM_SECRETS_MASTER_KEY", Fernet.generate_key().decode())

from app.core.config import get_settings
from app.core.deps import get_redis
from app.db.session import get_db
from app.main import create_app

get_settings.cache_clear()


@pytest.fixture
async def db_session() -> AsyncGenerator[AsyncSession, None]:
    engine = create_async_engine(
        str(get_settings().database_url),
        poolclass=NullPool,
    )
    session_factory = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)

    async with engine.connect() as connection:
        transaction = await connection.begin()
        session = session_factory(bind=connection)
        try:
            yield session
        finally:
            await session.close()
            await transaction.rollback()

    await engine.dispose()


@pytest.fixture
async def fake_redis() -> AsyncGenerator[Redis, None]:
    server = fakeredis.aioredis.FakeRedis(decode_responses=True)
    yield server
    await server.flushall()
    await server.aclose()


@pytest.fixture
async def client(db_session: AsyncSession, fake_redis: Redis) -> AsyncGenerator[AsyncClient, None]:
    app = create_app()

    async def override_get_db() -> AsyncGenerator[AsyncSession, None]:
        yield db_session

    async def override_get_redis() -> AsyncGenerator[Redis, None]:
        yield fake_redis

    app.dependency_overrides[get_db] = override_get_db
    app.dependency_overrides[get_redis] = override_get_redis

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        yield ac

    app.dependency_overrides.clear()


@pytest.fixture
def test_password() -> str:
    return "StrongPass123!"


@pytest.fixture
def unique_email() -> str:
    import uuid

    return f"user-{uuid.uuid4().hex[:8]}@example.com"
