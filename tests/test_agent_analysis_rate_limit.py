import pytest
from unittest.mock import AsyncMock, MagicMock
from uuid import uuid4

from app.services.agent_rate_limit import check_agent_analysis_limit


@pytest.mark.asyncio
async def test_analysis_rate_limit_allows_under_cap():
    redis = AsyncMock()
    redis.incr = AsyncMock(return_value=1)
    redis.expire = AsyncMock()
    settings = MagicMock()
    settings.agent_analysis_rate_limit_per_hour = 20
    settings.is_development = True
    await check_agent_analysis_limit(redis, settings, uuid4())
