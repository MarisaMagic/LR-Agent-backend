"""In-memory sessions for backend-driven sub-image ReAct runs (client executes local tools)."""
from __future__ import annotations

import asyncio
import time
import uuid
from dataclasses import dataclass, field
from typing import Any

SESSION_TTL_S = 900
TOOL_WAIT_TIMEOUT_S = 600

_CLIENT_TOOLS = frozenset({"run_object_detection", "finalize_image_change"})


@dataclass
class SubImageRunState:
    boxes: list[dict[str, Any]] = field(default_factory=list)
    raw_count: int = 0
    kept_count: int = 0
    excluded_count: int = 0
    mappings: list[dict[str, Any]] = field(default_factory=list)
    map_method: str = ""
    map_hint: str = ""
    captured_finalize: bool = False


@dataclass
class SubImageRunSession:
    run_id: str
    user_id: uuid.UUID
    created_at: float = field(default_factory=time.time)
    state: SubImageRunState = field(default_factory=SubImageRunState)
    _pending: dict[str, asyncio.Future[str]] = field(default_factory=dict)
    closed: bool = False

    async def wait_client_tool(self, tool_call_id: str) -> str:
        if self.closed:
            raise RuntimeError("session_closed")
        loop = asyncio.get_running_loop()
        fut: asyncio.Future[str] = loop.create_future()
        self._pending[tool_call_id] = fut
        try:
            return await asyncio.wait_for(fut, timeout=TOOL_WAIT_TIMEOUT_S)
        finally:
            self._pending.pop(tool_call_id, None)

    def submit_tool_result(self, tool_call_id: str, content: str) -> bool:
        fut = self._pending.get(tool_call_id)
        if fut is None or fut.done():
            return False
        fut.set_result(content)
        return True

    def close(self) -> None:
        self.closed = True
        for fut in self._pending.values():
            if not fut.done():
                fut.set_exception(RuntimeError("session_closed"))


_sessions: dict[str, SubImageRunSession] = {}
_lock = asyncio.Lock()


def is_client_tool(name: str) -> bool:
    return name in _CLIENT_TOOLS


async def create_session(user_id: uuid.UUID) -> SubImageRunSession:
    await _purge_expired()
    run_id = uuid.uuid4().hex
    session = SubImageRunSession(run_id=run_id, user_id=user_id)
    async with _lock:
        _sessions[run_id] = session
    return session


async def get_session(run_id: str, user_id: uuid.UUID) -> SubImageRunSession | None:
    async with _lock:
        session = _sessions.get(run_id)
    if session is None or session.user_id != user_id:
        return None
    if time.time() - session.created_at > SESSION_TTL_S:
        await remove_session(run_id)
        return None
    return session


async def remove_session(run_id: str) -> None:
    async with _lock:
        session = _sessions.pop(run_id, None)
    if session:
        session.close()


async def _purge_expired() -> None:
    now = time.time()
    async with _lock:
        expired = [rid for rid, s in _sessions.items() if now - s.created_at > SESSION_TTL_S]
        for rid in expired:
            sess = _sessions.pop(rid, None)
            if sess:
                sess.close()
