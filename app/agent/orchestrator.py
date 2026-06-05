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
from app.agent.turn_understanding_service import (
    TurnUnderstandingResult,
    format_understanding_for_system_prompt,
    understand_turn,
)
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

        client_ctx = req.client_context
        has_project_snapshot = bool(
            client_ctx and client_ctx.annotation_project_snapshot is not None
        )
        has_workspace = bool(
            client_ctx and (client_ctx.workspace_root or "").strip()
        )
        has_assist_tools = has_project_snapshot or has_workspace

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

        if has_assist_tools:
            system_prompt = build_assist_system_prompt(
                client_ctx,
                model=row.model,
                provider_label=row.name,
                supports_vision=provider_is_vision,
            )
        else:
            system_prompt = f"{identity}\n\n{CHAT_SYSTEM_PROMPT}"

        understanding: TurnUnderstandingResult | None = None
        if client_ctx and client_ctx.turn_understanding is not None:
            tu = client_ctx.turn_understanding
            understanding = TurnUnderstandingResult(
                resolved_user_content=tu.resolved_user_content or req.user_content,
                referenced_relative_paths=list(tu.referenced_relative_paths or []),
                resolved_active_relative_path=tu.resolved_active_relative_path,
                task_intent=tu.task_intent or "converse",
                turn_kind=tu.turn_kind,
                needs_vision_input=bool(tu.needs_vision_input),
                confidence=float(tu.confidence),
                scope_notes=tu.scope_notes or "",
                reason=tu.reason or "client",
                user_visible_hint=tu.user_visible_hint,
            )
        elif has_assist_tools and not await cancelled():
            try:
                understanding = await understand_turn(
                    llm,
                    user_content=req.user_content,
                    client_context=client_ctx,
                    conversation_transcript=turn_ctx.transcript,
                    provider_is_vision=provider_is_vision,
                )
            except Exception:
                logger.exception(
                    "turn understand failed session=%s user=%s",
                    req.session_id,
                    user.id,
                )

        if understanding is not None:
            system_prompt = (
                f"{system_prompt}\n\n{format_understanding_for_system_prompt(understanding)}"
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

            if not await cancelled():
                yield StreamEventPayload(
                    type="route_decided",
                    mode="assist" if has_assist_tools else "chat",
                    domain="annotation" if has_project_snapshot else "general",
                )

            if has_assist_tools:
                tools = build_tools_p1(
                    user,
                    req.client_context,
                    settings=self.settings,
                    provider_is_vision=provider_is_vision,
                )
                stream = assist_service.stream_assist(
                    llm,
                    lc_messages,
                    tools,
                    settings=self.settings,
                    max_tool_rounds=self.settings.agent_max_tool_rounds,
                    is_cancelled=cancelled,
                    provider_is_vision=provider_is_vision,
                    needs_vision_input=bool(
                        understanding and understanding.needs_vision_input
                    ),
                    client_context=client_ctx,
                    understanding=understanding,
                    user_content=req.user_content,
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
        except Exception:
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
