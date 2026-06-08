"""单张图子 Agent 运行的内存会话管理。

后端驱动 ReAct 循环时，客户端工具（YOLO 检测、finalize 写入）通过 Future 异步等待结果：
  1. stream_sub_image_run 发出 client_tool SSE 事件
  2. 客户端本地执行后 POST /sub-image-run/tool-result
  3. submit_tool_result 唤醒 wait_client_tool，循环继续

map_detection_boxes_to_labels 在服务端执行，不走此会话等待机制。
"""

from __future__ import annotations

import asyncio
import time
import uuid
from dataclasses import dataclass, field
from typing import Any

SESSION_TTL_S = 900
TOOL_WAIT_TIMEOUT_S = 600

# 需在客户端 Electron 本地执行的工具
_CLIENT_TOOLS = frozenset({"run_object_detection", "finalize_image_change"})


@dataclass
class SubImageRunState:
    """单张图标注过程中的累积状态（检测框、映射结果、是否已 finalize）。"""
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
    """单次 sub_image_run 的会话实例，绑定 user_id 与 run_id。"""
    run_id: str
    user_id: uuid.UUID
    created_at: float = field(default_factory=time.time)
    state: SubImageRunState = field(default_factory=SubImageRunState)
    _pending: dict[str, asyncio.Future[str]] = field(default_factory=dict)
    closed: bool = False

    async def wait_client_tool(self, tool_call_id: str) -> str:
        """挂起等待客户端提交工具执行结果，超时或会话关闭时抛异常。"""
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
        """由 API 端点调用，唤醒对应 tool_call 的等待 Future。"""
        fut = self._pending.get(tool_call_id)
        if fut is None or fut.done():
            return False
        fut.set_result(content)
        return True

    def close(self) -> None:
        """关闭会话，取消所有未完成的工具等待。"""
        self.closed = True
        for fut in self._pending.values():
            if not fut.done():
                fut.set_exception(RuntimeError("session_closed"))


_sessions: dict[str, SubImageRunSession] = {}
_lock = asyncio.Lock()


def is_client_tool(name: str) -> bool:
    """判断工具是否需在客户端本地执行。"""
    return name in _CLIENT_TOOLS


async def create_session(user_id: uuid.UUID) -> SubImageRunSession:
    """创建新会话并注册到全局字典，启动前清理过期会话。"""
    await _purge_expired()
    run_id = uuid.uuid4().hex
    session = SubImageRunSession(run_id=run_id, user_id=user_id)
    async with _lock:
        _sessions[run_id] = session
    return session


async def get_session(run_id: str, user_id: uuid.UUID) -> SubImageRunSession | None:
    """按 run_id 获取会话，校验 user_id 与 TTL。"""
    async with _lock:
        session = _sessions.get(run_id)
    if session is None or session.user_id != user_id:
        return None
    if time.time() - session.created_at > SESSION_TTL_S:
        await remove_session(run_id)
        return None
    return session


async def remove_session(run_id: str) -> None:
    """移除并关闭会话（stream 结束时在 finally 中调用）。"""
    async with _lock:
        session = _sessions.pop(run_id, None)
    if session:
        session.close()


async def _purge_expired() -> None:
    """清理超过 SESSION_TTL_S 的过期会话。"""
    now = time.time()
    async with _lock:
        expired = [rid for rid, s in _sessions.items() if now - s.created_at > SESSION_TTL_S]
        for rid in expired:
            sess = _sessions.pop(rid, None)
            if sess:
                sess.close()
