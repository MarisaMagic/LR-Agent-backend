"""Annotation batch turn persistence (align with chat PG + Redis)."""
from __future__ import annotations

import uuid

from redis.asyncio import Redis
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import Settings
from app.models.user import User
from app.schemas.agent import (
    AnnotationRunEventsRequest,
    AnnotationRunFinalizeRequest,
    AnnotationRunStartRequest,
    ChatStreamRequest,
    StreamEventPayload,
)
from app.services.agent_chat_repository import AgentChatRepository
from app.services.agent_job_service import AgentJobService
from app.services.llm_provider_service import LlmProviderService


class AnnotationRunService:
    def __init__(
        self,
        db: AsyncSession,
        redis: Redis,
        settings: Settings,
    ) -> None:
        self.db = db
        self.repo = AgentChatRepository(db, redis, settings)
        self.jobs = AgentJobService(redis, settings)
        self.settings = settings

    async def start_turn(
        self,
        user: User,
        body: AnnotationRunStartRequest,
    ) -> dict[str, str]:
        provider_svc = LlmProviderService(self.db, self.settings)
        provider_uuid = uuid.UUID(body.provider_id)
        row = await provider_svc.get_for_user(provider_uuid, user.id)
        if row is None or not row.enabled:
            raise ValueError("llm_provider_not_found")

        await self.jobs.register_job(str(user.id), body.client_job_id)

        req = ChatStreamRequest(
            provider_id=body.provider_id,
            session_id=body.session_id,
            client_job_id=body.client_job_id,
            user_content=body.user_content,
            user_message_id=body.user_message_id,
            assistant_message_id=body.assistant_message_id,
            truncate_from_message_id=body.truncate_from_message_id,
            client_context=body.client_context,
        )
        try:
            await self.repo.prepare_turn(
                user.id,
                req,
                user_message_id=body.user_message_id,
                assistant_message_id=body.assistant_message_id,
                provider_model=row.model,
            )
        except PermissionError as exc:
            raise ValueError("session_forbidden") from exc

        await self.repo.replay_pending_events(
            user.id, body.session_id, body.assistant_message_id,
        )

        return {
            "session_id": body.session_id,
            "assistant_message_id": body.assistant_message_id,
        }

    async def ingest_events(
        self,
        user: User,
        body: AnnotationRunEventsRequest,
    ) -> dict[str, bool]:
        session_row = await self.repo.get_session_for_user(user.id, body.session_id)
        if session_row is None:
            filtered: list[dict[str, Any]] = []
            for raw in body.events:
                if not isinstance(raw, dict):
                    continue
                event_type = str(raw.get("type") or "")
                if event_type in ("done", "error", "preparing", "route_decided", "context_updated"):
                    continue
                filtered.append(raw)
            if filtered:
                await self.repo.cache_pending_events(
                    user.id, body.session_id, body.assistant_message_id, filtered,
                )
            return {"ok": True, "pending": True}

        if not await self.repo.try_advance_event_seq(user.id, body.client_job_id, body.seq):
            return {"ok": True, "duplicate": True}

        payloads: list[StreamEventPayload] = []
        for raw in body.events:
            if not isinstance(raw, dict):
                continue
            event_type = str(raw.get("type") or "")
            if event_type in ("done", "error", "preparing", "route_decided", "context_updated"):
                continue
            payloads.append(StreamEventPayload.from_client_dict(raw))

        if payloads:
            await self.repo.apply_stream_events_batch(
                user.id,
                body.session_id,
                body.assistant_message_id,
                payloads,
            )
        return {"ok": True, "duplicate": False}

    async def finalize_turn(
        self,
        user: User,
        body: AnnotationRunFinalizeRequest,
    ) -> dict[str, bool]:
        session_row = await self.repo.get_session_for_user(user.id, body.session_id)
        if session_row is None:
            raise ValueError("session_not_found")

        await self.repo.finalize_annotation_turn(
            user.id,
            body.session_id,
            body.assistant_message_id,
            status=body.status,
            error=body.error,
            user_content=body.user_content,
        )
        return {"ok": True}
