import logging
import uuid
from collections.abc import AsyncIterator

from redis.asyncio import Redis
from sqlalchemy.ext.asyncio import AsyncSession

from app.agent import assist_service, chat_service, context_service
from app.agent.context_service import CHAT_SYSTEM_PROMPT, build_lc_messages
from app.agent.context_snapshot import build_ask_system_prompt
from app.agent.llm_factory import build_chat_model
from app.agent.router_service import RouteDecision, classify_intent
from app.agent.tools.registry import build_tools_p1
from app.core.config import Settings
from app.models.user import User
from app.schemas.agent import ChatContextInput, ChatStreamRequest, StreamEventPayload
from app.services.agent_chat_repository import AgentChatRepository
from app.services.agent_job_service import AgentJobService
from app.services.llm_provider_service import LlmProviderService

logger = logging.getLogger(__name__)


class ChatOrchestrator:
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
        user_id_str = str(user.id)
        provider_svc = LlmProviderService(self.db, self.settings)
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

        await self.jobs.register_job(user_id_str, req.client_job_id)

        api_key = provider_svc.decrypt_api_key(row)
        llm = build_chat_model(row, api_key)

        cancelled = lambda: self.jobs.is_cancelled(user_id_str, req.client_job_id)
        assistant_id = req.assistant_message_id

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

        req.messages = chat_inputs
        if req.context is None:
            req.context = ChatContextInput()
        req.context.summary = session_row.context_summary
        req.context.summary_up_to_message_id = session_row.summary_up_to_message_id

        yield StreamEventPayload(type="preparing", stage="build_messages")

        client_ctx = req.client_context
        has_project_snapshot = bool(
            client_ctx and client_ctx.annotation_project_snapshot is not None
        )
        chat_mode = not client_ctx or client_ctx.agent_mode in (
            None,
            "chat",
            "ask",
        )
        ask_mode = chat_mode
        system_prompt = (
            build_ask_system_prompt(client_ctx)
            if has_project_snapshot
            else CHAT_SYSTEM_PROMPT
        )

        lc_messages, token_estimate, needs_summarize = build_lc_messages(
            req,
            self.settings,
            system_prompt=system_prompt,
        )

        final_status = "done"
        error_message: str | None = None

        try:
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

            route = RouteDecision(
                mode="chat",
                confidence=1.0,
                domain="general",
                reason="router_disabled",
            )
            if self.settings.agent_router_enabled and not await cancelled() and not ask_mode:
                try:
                    route = await classify_intent(llm, req.user_content)
                except Exception:
                    route = RouteDecision(
                        mode="chat",
                        confidence=0.5,
                        domain="general",
                        reason="router_failed",
                    )
            elif ask_mode and has_project_snapshot:
                route = RouteDecision(
                    mode="assist",
                    confidence=1.0,
                    domain="annotation",
                    reason="ask_mode_with_project",
                )
            if not await cancelled():
                yield StreamEventPayload(
                    type="route_decided",
                    mode=route.mode,
                    domain=route.domain,
                )

            use_assist = route.mode == "assist" or (ask_mode and has_project_snapshot)
            if use_assist:
                lc_messages, _, _ = build_lc_messages(
                    req,
                    self.settings,
                    system_prompt=system_prompt,
                )
                tools = build_tools_p1(user, req.client_context)
                stream = assist_service.stream_assist(
                    llm,
                    lc_messages,
                    tools,
                    max_tool_rounds=self.settings.agent_max_tool_rounds,
                    is_cancelled=cancelled,
                )
            else:
                stream = chat_service.stream_chat(llm, lc_messages)

            async for event in self._persist_stream(
                user.id,
                req.session_id,
                assistant_id,
                stream,
                is_cancelled=cancelled,
            ):
                yield event

            if await cancelled():
                final_status = "stopped"
        except Exception as exc:
            final_status = "error"
            error_message = "stream_failed"
            logger.exception("chat stream failed session=%s user=%s", req.session_id, user.id)
            yield StreamEventPayload(type="error", message=error_message)
        finally:
            if await cancelled() and final_status == "done":
                final_status = "stopped"
            await self.repo.finalize_assistant_message(
                user.id,
                req.session_id,
                assistant_id,
                status=final_status,
                error=error_message,
            )

        if not await cancelled() and final_status == "done":
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
        async for event in source:
            if await is_cancelled():
                return
            if event.type in (
                "text_delta",
                "reasoning_delta",
                "tool_start",
                "tool_result",
            ):
                await self.repo.apply_stream_event(
                    user_id,
                    session_id,
                    assistant_message_id,
                    event,
                )
            yield event
