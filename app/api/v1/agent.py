"""Agent API — stateless Assist chat/stream for tool-enabled sessions.

Session/chat CRUD and LLM provider config live in the Electron frontend
(local SQLite storage + direct LLM API calls for simple chat).
"""

import asyncio
import json
import logging
from typing import Any

from fastapi import APIRouter
from fastapi.responses import StreamingResponse
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

from app.agent import assist_service, chat_service
from app.agent.assist_mode_router import FULL_TOOL_SET, LIGHT_TOOL_SET
from app.agent.context_service import CHAT_SYSTEM_PROMPT
from app.agent.context_snapshot import (
    build_assist_system_prompt,
    format_runtime_identity_block,
)
from app.agent.tools.mcp_client import load_mcp_tools_from_server
from app.agent.tools.registry import build_tools_by_name_set
from app.agent.tools.tool_registry_meta import CANONICAL_CAPABILITIES
from app.core.deps import SettingsDep
from app.models.user import User
from app.schemas.agent import ChatCancelRequest, ClientContextInput, LocalChatStreamRequest

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/agent", tags=["agent"])

_cancel_events: dict[str, asyncio.Event] = {}


def _get_or_create_cancel_event(client_job_id: str) -> asyncio.Event:
    if client_job_id not in _cancel_events:
        _cancel_events[client_job_id] = asyncio.Event()
    return _cancel_events[client_job_id]


def _cleanup_cancel_event(client_job_id: str) -> None:
    _cancel_events.pop(client_job_id, None)


def _anonymous_user() -> User:
    """Minimal anonymous User for tool building in stateless mode."""
    import uuid

    return User(
        id=uuid.uuid4(),
        email="anonymous@local",
        email_verified=False,
        username="anonymous",
        display_name="本地匿名用户",
    )


def _build_lc_messages_from_local(
    body: LocalChatStreamRequest,
    system_prompt: str,
) -> list:
    """Build LangChain messages from LocalChatStreamRequest (no DB dependency)."""
    lc_messages: list = [SystemMessage(content=system_prompt)]
    if body.context_summary:
        lc_messages.append(
            SystemMessage(content=f"【此前对话摘要】\n{body.context_summary}"),
        )
    for item in body.messages:
        if item.role == "user":
            lc_messages.append(HumanMessage(content=item.content))
        elif item.role == "assistant":
            lc_messages.append(AIMessage(content=item.content))
        elif item.role == "system":
            lc_messages.append(SystemMessage(content=item.content))
    return lc_messages


async def _stream_local_chat(body: LocalChatStreamRequest, settings) -> Any:
    """无状态 chat/stream 生成器——所有数据从请求体获取，不读 DB。"""
    client_job_id: str = body.client_job_id
    cancel_event = _get_or_create_cancel_event(client_job_id)

    async def is_cancelled() -> bool:
        return cancel_event.is_set()

    try:
        base_url = body.base_url.rstrip("/")
        llm = ChatOpenAI(
            model=body.model,
            api_key=body.api_key,
            base_url=base_url,
            streaming=True,
            temperature=0.7,
            timeout=120,
        )

        client_ctx: ClientContextInput | None = body.client_context
        has_tools = bool(
            client_ctx
            and (
                (client_ctx.workspace_root or "").strip()
                or client_ctx.active_annotation_project_id
                or client_ctx.annotation_project_snapshot is not None
            )
        )

        if has_tools:
            user = _anonymous_user()

            identity = format_runtime_identity_block(
                model=body.model,
                provider_label="local",
                supports_vision=body.supports_vision,
            )
            if client_ctx:
                system_prompt = build_assist_system_prompt(
                    client_ctx,
                    model=body.model,
                    provider_label="local",
                    supports_vision=body.supports_vision,
                )
            else:
                system_prompt = f"{identity}\n\n{CHAT_SYSTEM_PROMPT}"

            lc_messages = _build_lc_messages_from_local(body, system_prompt)

            has_workspace = bool(client_ctx and (client_ctx.workspace_root or "").strip())
            has_project_snapshot = bool(
                client_ctx and client_ctx.annotation_project_snapshot is not None
            )
            is_editor = bool(client_ctx and client_ctx.work_mode == "editor")

            if has_project_snapshot:
                tool_set = FULL_TOOL_SET
            elif is_editor or has_workspace:
                tool_set = LIGHT_TOOL_SET
            else:
                tool_set = frozenset()
            tools = build_tools_by_name_set(
                user,
                client_ctx,
                tool_set,
                settings=settings,
                provider_is_vision=body.supports_vision,
            )

            mcp_url = (
                (client_ctx.mcp_server_url or "").strip()
                if client_ctx
                else ""
            )
            if mcp_url:
                try:
                    mcp_tools = await load_mcp_tools_from_server(
                        mcp_url,
                        existing_capabilities=CANONICAL_CAPABILITIES,
                    )
                    if mcp_tools:
                        existing = {t.name for t in tools}
                        tools = tools + [
                            t for t in mcp_tools if t.name not in existing
                        ]
                except Exception:
                    logger.warning(
                        "Failed to load MCP tools from %s",
                        mcp_url,
                        exc_info=True,
                    )

            if body.client_tool_results:
                assist_service.append_client_tool_results_to_messages(
                    lc_messages,
                    body.client_tool_results,
                    user_content=body.user_content,
                )

            stream = assist_service.stream_assist(
                llm,
                lc_messages,
                tools,
                settings=settings,
                max_tool_rounds=settings.agent_max_tool_rounds,
                is_cancelled=is_cancelled,
                provider_is_vision=body.supports_vision,
                client_context=client_ctx,
                user_content=body.user_content,
                client_tool_results=body.client_tool_results or None,
            )
        else:
            system_prompt = body.system_prompt or CHAT_SYSTEM_PROMPT
            lc_messages = _build_lc_messages_from_local(body, system_prompt)
            stream = chat_service.stream_chat(llm, lc_messages)

        async for event in stream:
            if await is_cancelled():
                break
            yield f"data: {json.dumps(event.to_sse_dict(), ensure_ascii=False)}\n\n"

        yield 'data: {"type": "done"}\n\n'

    finally:
        _cleanup_cancel_event(client_job_id)


@router.post("/chat/stream")
async def local_chat_stream(
    body: LocalChatStreamRequest,
    settings: SettingsDep,
) -> StreamingResponse:
    """无状态 chat/stream —— 所有数据从请求体获取，不读 DB。"""
    return StreamingResponse(
        _stream_local_chat(body, settings),
        media_type="text/event-stream",
    )


@router.post("/chat/cancel")
async def cancel_chat(
    body: ChatCancelRequest,
) -> dict[str, bool]:
    """取消正在进行的 chat/stream 任务。"""
    event = _cancel_events.get(body.client_job_id)
    if event:
        event.set()
    return {"ok": True}
