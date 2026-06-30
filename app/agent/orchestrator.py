"""聊天 Agent 编排器：串联单次对话轮次的完整生命周期。

编排流程概览：
  1. 校验 LLM 提供商 → 注册客户端任务 → 构建模型实例
  2. 同步上下文、准备本轮消息 → 构建轮次上下文窗口
  3. 探测视觉能力 → 组装系统提示词
  4. 构建 LangChain 消息 → 按需压缩历史摘要
  5. 路由到 assist（工具调用）或 chat（纯对话）并流式输出
  6. 持久化流事件 → 落库助手消息终态
"""

import logging
import uuid
from collections.abc import AsyncIterator

from redis.asyncio import Redis
from sqlalchemy.ext.asyncio import AsyncSession

from app.agent import assist_service, chat_service, context_service
from app.agent.context_service import CHAT_SYSTEM_PROMPT, build_lc_messages
from app.agent.context_snapshot import (
    build_assist_system_prompt,
    format_runtime_identity_block,
)
from app.agent.llm_factory import build_chat_model
from app.agent.turn_context import build_turn_context_from_messages, turn_context_to_chat_inputs
from app.agent.tools.mcp_client import load_mcp_tools_from_server
from app.agent.tools.tool_registry_meta import CANONICAL_CAPABILITIES
from app.agent.assist_mode_router import AssistMode, AssistModeRouter
from app.agent.tools.registry import build_tools_by_name_set, build_tools_p1
from app.core.config import Settings
from app.models.user import User
from app.schemas.agent import ChatContextInput, ChatStreamRequest, JobState, StreamEventPayload
from app.services.agent_chat_blocks import PERSISTABLE_STREAM_EVENT_TYPES
from app.services.agent_chat_repository import AgentChatRepository
from app.services.agent_job_service import AgentJobService
from app.services.llm_provider_service import LlmProviderService

logger = logging.getLogger(__name__)


class ChatOrchestrator:
    """单次聊天请求的编排入口，协调仓储、上下文、理解与流式推理。"""

    def __init__(
        self,
        db: AsyncSession,
        redis: Redis,
        redis_jobs: AgentJobService,
        settings: Settings,
    ) -> None:
        self.db = db
        self.repo = AgentChatRepository(db, redis, settings)
        self.jobs = redis_jobs
        self.settings = settings

    async def run(
        self,
        user: User,
        req: ChatStreamRequest,
    ) -> AsyncIterator[StreamEventPayload]:
        """执行一轮对话，以 SSE 事件流形式逐步产出 preparing / delta / done 等事件。"""
        user_id_str = str(user.id)
        provider_svc = LlmProviderService(self.db, self.settings)

        # ── 阶段 1：校验 LLM 提供商 ──────────────────────────────────────
        try:
            provider_uuid = uuid.UUID(req.provider_id)
        except ValueError:
            yield StreamEventPayload(type="error", message="invalid_provider_id")
            return

        row = await provider_svc.get_for_user(provider_uuid, user.id)
        if row is None or not row.enabled:
            yield StreamEventPayload(type="error", message="llm_provider_not_found")
            return

        try:
            provider_svc.validate_provider_base_url(row)
        except ValueError:
            yield StreamEventPayload(type="error", message="invalid_base_url")
            return

        # ── 阶段 2：注册任务、构建模型 ────────────────────────────────────
        await self.jobs.register_job(user_id_str, req.client_job_id)

        api_key = provider_svc.decrypt_api_key(row)
        llm = build_chat_model(row, api_key)

        cancelled = lambda: self.jobs.is_cancelled(user_id_str, req.client_job_id)
        assistant_id = req.assistant_message_id

        # ── 阶段 3：同步上下文、准备本轮消息 ──────────────────────────────
        await self.repo.sync_context_from_request(user.id, req)

        try:
            chat_inputs, session_row = await self.repo.prepare_turn(
                user.id,
                req,
                user_message_id=req.user_message_id,
                assistant_message_id=assistant_id,
                provider_model=row.model,
            )
        except PermissionError:
            yield StreamEventPayload(type="error", message="session_forbidden")
            return
        except Exception:
            logger.exception("prepare_turn failed session=%s user=%s", req.session_id, user.id)
            yield StreamEventPayload(type="error", message="prepare_turn_failed")
            return

        await self.repo.replay_pending_events(
            user.id, req.session_id, assistant_id,
        )

        # 裁剪对话窗口，排除尚未完成的助手占位消息
        turn_ctx = build_turn_context_from_messages(
            chat_inputs,
            current_user_content=req.user_content,
            summary=session_row.context_summary,
            summary_up_to_message_id=session_row.summary_up_to_message_id,
            max_turns_in_window=(
                req.context.config.max_turns_in_window
                if req.context and req.context.config
                else self.settings.agent_default_max_turns_in_window
            ),
            exclude_message_ids={assistant_id},
        )
        req.messages = turn_context_to_chat_inputs(turn_ctx)
        if req.context is None:
            req.context = ChatContextInput()
        req.context.summary = session_row.context_summary
        req.context.summary_up_to_message_id = session_row.summary_up_to_message_id

        yield StreamEventPayload(type="preparing", stage="build_messages")

        # ── 阶段 4：turn_kind 驱动路由 ─────────────────────────────────
        client_ctx = req.client_context
        has_project_snapshot = bool(
            client_ctx and client_ctx.annotation_project_snapshot is not None
        )
        has_workspace = bool(
            client_ctx and (client_ctx.workspace_root or "").strip()
        )

        # 探测模型是否支持视觉输入，供后续提示词与工具路由使用
        provider_is_vision = False
        if not await cancelled():
            try:
                provider_is_vision = await provider_svc.ensure_vision_probed(row)
            except Exception:
                logger.exception(
                    "vision probe failed session=%s user=%s",
                    req.session_id,
                    user.id,
                )

        identity = format_runtime_identity_block(
            model=row.model,
            provider_label=row.name,
            supports_vision=provider_is_vision,
        )

        # 使用 turn_kind 驱动路由决策
        turn_kind = (
            client_ctx.turn_understanding.turn_kind
            if client_ctx and client_ctx.turn_understanding
            else "converse"
        )
        assist_mode, tool_set = AssistModeRouter.resolve(
            turn_kind,
            has_project_snapshot=has_project_snapshot,
            has_workspace=has_workspace,
        )
        has_assist_tools = assist_mode != AssistMode.CHAT

        if assist_mode == AssistMode.CHAT:
            system_prompt = f"{identity}\n\n{CHAT_SYSTEM_PROMPT}"
        else:
            system_prompt = build_assist_system_prompt(
                client_ctx,
                model=row.model,
                provider_label=row.name,
                supports_vision=provider_is_vision,
            )

        # ── 阶段 6：构建 LangChain 消息，评估是否需要压缩 ─────────────────
        lc_messages, token_estimate, needs_summarize = build_lc_messages(
            req,
            self.settings,
            system_prompt=system_prompt,
        )

        final_status = "done"
        error_message: str | None = None
        tool_pending_emitted = False

        try:
            # 上下文超长时先压缩历史，再重新构建消息
            if needs_summarize and not await cancelled():
                yield StreamEventPayload(type="preparing", stage="summarize")
                summary = await chat_service.summarize_messages(llm, req.messages)
                summary_msg_id = (
                    req.truncate_from_message_id
                    or (req.context.summary_up_to_message_id if req.context else None)
                    or req.user_message_id
                )
                yield StreamEventPayload(
                    type="context_updated",
                    summary=summary,
                    summary_up_to_message_id=summary_msg_id,
                    token_estimate=token_estimate,
                )
                await self.repo.update_session_summary(
                    user.id,
                    req.session_id,
                    summary=summary,
                    summary_up_to_message_id=summary_msg_id,
                    token_estimate=token_estimate,
                )
                if req.context is None:
                    req.context = ChatContextInput(summary=summary)
                else:
                    req.context.summary = summary
                lc_messages, token_estimate, _ = build_lc_messages(
                    req,
                    self.settings,
                    system_prompt=system_prompt,
                )

            if not await cancelled():
                yield StreamEventPayload(
                    type="route_decided",
                    mode=assist_mode.value if has_assist_tools else "chat",
                    domain="annotation" if has_project_snapshot else "general",
                )

            # ── 阶段 7：路由并流式推理 ──────────────────────────────────────
            if has_assist_tools:
                # 根据 AssisModeRouter 决议构建工具集
                if assist_mode == AssistMode.FULL:
                    tools = build_tools_p1(
                        user,
                        req.client_context,
                        settings=self.settings,
                        provider_is_vision=provider_is_vision,
                    )
                else:
                    tools = build_tools_by_name_set(
                        user,
                        req.client_context,
                        tool_set=tool_set,
                        settings=self.settings,
                        provider_is_vision=provider_is_vision,
                    )
                # ── MCP 工具动态注入（Phase 3）─────────────────────────────
                mcp_url = (
                    (req.client_context.mcp_server_url or "").strip()
                    if req.client_context
                    else ""
                )
                if mcp_url:
                    mcp_tools = await load_mcp_tools_from_server(
                        mcp_url,
                        existing_capabilities=CANONICAL_CAPABILITIES,
                    )
                    if mcp_tools:
                        existing = {t.name for t in tools}
                        tools = tools + [
                            t for t in mcp_tools if t.name not in existing
                        ]
                stream = assist_service.stream_assist(
                    llm,
                    lc_messages,
                    tools,
                    settings=self.settings,
                    max_tool_rounds=self.settings.agent_max_tool_rounds,
                    is_cancelled=cancelled,
                    provider_is_vision=provider_is_vision,
                    client_context=client_ctx,
                    user_content=req.user_content,
                    client_tool_results=req.client_tool_results or None,
                )
            else:
                stream = chat_service.stream_chat(llm, lc_messages)

            # 根据是否为 resume 设置初始状态
            if req.client_tool_results:
                await self.jobs.set_state(user_id_str, req.client_job_id, JobState.RESUMING)
            else:
                await self.jobs.set_state(user_id_str, req.client_job_id, JobState.STREAMING)

            async for event in self._persist_stream(
                user.id,
                req.session_id,
                assistant_id,
                stream,
                is_cancelled=cancelled,
            ):
                if event.type == "tool_pending":
                    tool_pending_emitted = True
                    await self.jobs.set_state(user_id_str, req.client_job_id, JobState.TOOL_PENDING)
                yield event

            if await cancelled():
                final_status = "stopped"
            elif tool_pending_emitted:
                # 前端将 resume，保留消息为 streaming 态（等待 resume 后再 finalize）
                final_status = "streaming"
        except Exception:
            final_status = "error"
            error_message = "stream_failed"
            logger.exception("chat stream failed session=%s user=%s", req.session_id, user.id)
            yield StreamEventPayload(type="error", message=error_message)
        finally:
            # ── 阶段 8：落库助手消息终态 ────────────────────────────────────
            if await cancelled() and final_status == "done":
                final_status = "stopped"
            # 同步最终状态到 Redis
            if await cancelled():
                await self.jobs.set_state(user_id_str, req.client_job_id, JobState.CANCELLED)
            elif final_status == "error":
                await self.jobs.set_state(user_id_str, req.client_job_id, JobState.ERROR)
            elif final_status == "done":
                await self.jobs.set_state(user_id_str, req.client_job_id, JobState.DONE)
            # tool_pending 时保留 streaming 态，等前端 resume 后再 finalize
            if final_status != "streaming":
                await self.repo.finalize_assistant_message(
                    user.id,
                    req.session_id,
                    assistant_id,
                    status=final_status,
                    error=error_message,
                )

        if not await cancelled() and final_status == "done" and not tool_pending_emitted:
            yield StreamEventPayload(type="done")

    async def _persist_stream(
        self,
        user_id: uuid.UUID,
        session_id: str,
        assistant_message_id: str,
        source: AsyncIterator[StreamEventPayload],
        *,
        is_cancelled,
    ) -> AsyncIterator[StreamEventPayload]:
        """透传流事件，同时将文本/推理/工具事件增量写入仓储。"""
        async for event in source:
            if await is_cancelled():
                return
            if event.type in PERSISTABLE_STREAM_EVENT_TYPES:
                await self.repo.apply_stream_event(
                    user_id,
                    session_id,
                    assistant_message_id,
                    event,
                )
            yield event
