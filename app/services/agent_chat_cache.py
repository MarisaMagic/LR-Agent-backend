import json
from datetime import datetime, timezone
from typing import Any

from redis.asyncio import Redis

from app.core.config import Settings


class AgentChatCache:
    def __init__(self, redis: Redis, settings: Settings) -> None:
        self.redis = redis
        self.settings = settings
        self._ttl = settings.agent_chat_cache_ttl_seconds

    def _user_sessions_key(self, user_id: str) -> str:
        return f"agent:u:{user_id}:sessions"

    def _session_meta_key(self, user_id: str, session_id: str) -> str:
        return f"agent:u:{user_id}:s:{session_id}:meta"

    def _session_msg_ids_key(self, user_id: str, session_id: str) -> str:
        return f"agent:u:{user_id}:s:{session_id}:msg_ids"

    def _message_key(self, user_id: str, message_id: str) -> str:
        return f"agent:u:{user_id}:m:{message_id}"

    async def _touch_ttl(self, *keys: str) -> None:
        if not keys:
            return
        pipe = self.redis.pipeline()
        for key in keys:
            pipe.expire(key, self._ttl)
        await pipe.execute()

    async def invalidate_session(self, user_id: str, session_id: str) -> None:
        msg_ids = await self.redis.lrange(
            self._session_msg_ids_key(user_id, session_id),
            0,
            -1,
        )
        keys = [
            self._session_meta_key(user_id, session_id),
            self._session_msg_ids_key(user_id, session_id),
            *[self._message_key(user_id, mid) for mid in msg_ids],
        ]
        if keys:
            await self.redis.delete(*keys)
        await self.redis.zrem(self._user_sessions_key(user_id), session_id)

    async def set_session_meta(
        self,
        user_id: str,
        session_id: str,
        meta: dict[str, Any],
    ) -> None:
        key = self._session_meta_key(user_id, session_id)
        mapping = {
            k: json.dumps(v, ensure_ascii=False) if isinstance(v, (dict, list)) else str(v)
            for k, v in meta.items()
        }
        await self.redis.hset(key, mapping=mapping)
        updated_ms = int(meta.get("updated_at_ms") or datetime.now(timezone.utc).timestamp() * 1000)
        await self.redis.zadd(self._user_sessions_key(user_id), {session_id: updated_ms})
        limit = self.settings.agent_chat_sessions_index_limit
        await self.redis.zremrangebyrank(self._user_sessions_key(user_id), 0, -(limit + 1))
        await self._touch_ttl(key, self._user_sessions_key(user_id))

    async def get_session_meta(self, user_id: str, session_id: str) -> dict[str, Any] | None:
        raw = await self.redis.hgetall(self._session_meta_key(user_id, session_id))
        if not raw:
            return None
        return self._decode_meta(raw)

    def _decode_meta(self, raw: dict[str, str]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in raw.items():
            if value == "":
                result[key] = None
                continue
            if key in ("context_summary", "title", "provider_id", "model", "summary_up_to_message_id"):
                result[key] = value
                continue
            if key.endswith("_ms") or key == "last_context_token_estimate":
                try:
                    result[key] = int(value)
                except ValueError:
                    result[key] = value
                continue
            try:
                result[key] = json.loads(value)
            except json.JSONDecodeError:
                result[key] = value
        return result

    async def set_message_ids(self, user_id: str, session_id: str, message_ids: list[str]) -> None:
        key = self._session_msg_ids_key(user_id, session_id)
        pipe = self.redis.pipeline()
        pipe.delete(key)
        if message_ids:
            pipe.rpush(key, *message_ids)
        await pipe.execute()
        await self._touch_ttl(key)

    async def get_message_ids(self, user_id: str, session_id: str) -> list[str] | None:
        key = self._session_msg_ids_key(user_id, session_id)
        exists = await self.redis.exists(key)
        if not exists:
            return None
        return await self.redis.lrange(key, 0, -1)

    async def set_message(self, user_id: str, session_id: str, payload: dict[str, Any]) -> None:
        message_id = str(payload["id"])
        key = self._message_key(user_id, message_id)
        await self.redis.set(key, json.dumps(payload, ensure_ascii=False))
        await self._touch_ttl(key, self._session_msg_ids_key(user_id, session_id))

    async def get_message(self, user_id: str, message_id: str) -> dict[str, Any] | None:
        raw = await self.redis.get(self._message_key(user_id, message_id))
        if not raw:
            return None
        return json.loads(raw)

    async def get_messages_for_session(
        self,
        user_id: str,
        session_id: str,
    ) -> list[dict[str, Any]] | None:
        msg_ids = await self.get_message_ids(user_id, session_id)
        if msg_ids is None:
            return None
        if not msg_ids:
            return []
        pipe = self.redis.pipeline()
        for mid in msg_ids:
            pipe.get(self._message_key(user_id, mid))
        rows = await pipe.execute()
        messages: list[dict[str, Any]] = []
        for raw in rows:
            if raw:
                messages.append(json.loads(raw))
        return messages

    async def list_session_ids(self, user_id: str, *, limit: int = 100) -> list[str] | None:
        key = self._user_sessions_key(user_id)
        exists = await self.redis.exists(key)
        if not exists:
            return None
        return await self.redis.zrevrange(key, 0, max(0, limit - 1))
