import logging
import uuid
from datetime import datetime, timezone
from typing import Any

logger = logging.getLogger(__name__)

from redis.asyncio import Redis
from sqlalchemy import delete, func, select, tuple_, update
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import Settings
from app.models.agent_message import AgentMessage
from app.models.agent_session import AgentSession
from app.schemas.agent import (
    AgentMessagePublic,
    AgentSessionDetailResponse,
    AgentSessionPublic,
    ChatMessageInput,
    ChatStreamRequest,
    StreamEventPayload,
)
from app.services.agent_chat_blocks import (
    apply_stream_event_to_blocks,
    blocks_to_preview,
    blocks_to_text,
    collapse_assistant_blocks,
)
from app.services.agent_chat_cache import AgentChatCache
from app.services.agent_session_cursor import decode_session_cursor, encode_session_cursor


def normalize_message_interaction_mode(agent_mode: str | None) -> str:
    """Persisted per-message mode: chat (Ask) or annotation (Agent)."""
    if agent_mode in ("annotation", "annotate"):
        return "annotation"
    return "chat"


def _dt_to_ms(value: datetime | None) -> int:
    if value is None:
        return 0
    return int(value.timestamp() * 1000)


def _build_session_title(content: str) -> str:
    line = " ".join(content.strip().split())
    if not line:
        return "新对话"
    return f"{line[:24]}…" if len(line) > 24 else line


def _build_message_preview(blocks_json: list[dict[str, Any]], *, max_len: int = 128) -> str:
    return blocks_to_preview(blocks_json, max_len=max_len)


def _annotation_title_from_blocks(blocks_json: list[dict[str, Any]]) -> str | None:
    for block in blocks_json:
        if block.get("type") != "annotation_proposal":
            continue
        proposal = block.get("proposal")
        if not isinstance(proposal, dict):
            continue
        summary = str(proposal.get("summary") or "").strip()
        if summary:
            return _build_session_title(summary)
        stats = proposal.get("stats")
        if isinstance(stats, dict):
            succeeded = int(stats.get("succeeded") or 0)
            total_boxes = int(stats.get("totalBoxes") or stats.get("total_boxes") or 0)
            if succeeded:
                base = f"批量标注 {succeeded} 张"
                if total_boxes:
                    return f"{base} · {total_boxes} 框"
                return base
    return None


class AgentChatRepository:
    def __init__(self, db: AsyncSession, redis: Redis, settings: Settings) -> None:
        self.db = db
        self.cache = AgentChatCache(redis, settings)
        self.settings = settings

    async def get_session_for_user(
        self,
        user_id: uuid.UUID,
        session_id: str,
    ) -> AgentSession | None:
        result = await self.db.execute(
            select(AgentSession).where(
                AgentSession.id == session_id,
                AgentSession.user_id == user_id,
                AgentSession.deleted_at.is_(None),
            ),
        )
        return result.scalar_one_or_none()

    async def list_sessions(
        self,
        user_id: uuid.UUID,
        *,
        limit: int = 50,
        cursor: str | None = None,
        annotation_project_id: str | None = None,
        workspace_only: bool = False,
    ) -> tuple[list[AgentSessionPublic], str | None, bool]:
        stmt = select(AgentSession).where(
            AgentSession.user_id == user_id,
            AgentSession.deleted_at.is_(None),
        )
        if workspace_only:
            stmt = stmt.where(AgentSession.annotation_project_id.is_(None))
        elif annotation_project_id is not None:
            stmt = stmt.where(AgentSession.annotation_project_id == annotation_project_id)
        if cursor:
            try:
                cursor_dt, cursor_id = decode_session_cursor(cursor)
            except ValueError as exc:
                raise ValueError("invalid_cursor") from exc
            stmt = stmt.where(
                tuple_(AgentSession.updated_at, AgentSession.id)
                < tuple_(cursor_dt, cursor_id),
            )

        result = await self.db.execute(
            stmt.order_by(AgentSession.updated_at.desc(), AgentSession.id.desc()).limit(limit + 1),
        )
        rows = list(result.scalars().all())
        has_more = len(rows) > limit
        page_rows = rows[:limit]

        summaries = await self._fetch_session_summaries([row.id for row in page_rows])
        out: list[AgentSessionPublic] = []
        for row in page_rows:
            count, preview = summaries.get(row.id, (0, None))
            if count <= 0:
                continue
            out.append(self._row_to_session_summary_public(row, count, preview))

        next_cursor: str | None = None
        if has_more and page_rows:
            last = page_rows[-1]
            next_cursor = encode_session_cursor(_dt_to_ms(last.updated_at), last.id)

        return out, next_cursor, has_more

    async def get_session_detail(
        self,
        user_id: uuid.UUID,
        session_id: str,
        *,
        limit: int = 50,
        before_message_id: str | None = None,
    ) -> AgentSessionDetailResponse | None:
        row = await self.get_session_for_user(user_id, session_id)
        if row is None:
            return None

        total_count = await self._count_messages(session_id)
        messages, has_more_before = await self._load_messages_page(
            session_id,
            user_id=user_id,
            limit=limit,
            before_message_id=before_message_id,
        )
        count, preview = (total_count, None)
        if not before_message_id:
            summaries = await self._fetch_session_summaries([session_id])
            count, preview = summaries.get(session_id, (total_count, None))

        session_public = self._row_to_session_summary_public(row, count, preview).model_copy(
            update={"message_ids": [message.id for message in messages]},
        )
        return AgentSessionDetailResponse(
            session=session_public,
            messages=messages,
            has_more_before=has_more_before,
        )

    async def create_session(
        self,
        user_id: uuid.UUID,
        *,
        session_id: str,
        title: str = "新对话",
        provider_id: str | None = None,
        model: str | None = None,
        annotation_project_id: str | None = None,
        interaction_mode: str | None = None,
    ) -> AgentSessionPublic:
        now = datetime.now(timezone.utc)
        row = AgentSession(
            id=session_id,
            user_id=user_id,
            title=title,
            annotation_project_id=annotation_project_id,
            interaction_mode=interaction_mode,
            provider_id=provider_id,
            model=model,
            created_at=now,
            updated_at=now,
        )
        self.db.add(row)
        await self.db.flush()
        public = self._row_to_session_summary_public(row, 0, None)
        await self._warm_session_meta(str(user_id), row, message_ids=[])
        return public

    async def update_session(
        self,
        user_id: uuid.UUID,
        session_id: str,
        *,
        title: str | None = None,
        provider_id: str | None = None,
        model: str | None = None,
    ) -> AgentSessionPublic | None:
        row = await self.get_session_for_user(user_id, session_id)
        if row is None:
            return None
        if title is not None:
            row.title = title
        if provider_id is not None:
            row.provider_id = provider_id
        if model is not None:
            row.model = model
        row.updated_at = datetime.now(timezone.utc)
        await self.db.flush()
        msg_ids = await self._message_ids_from_db(session_id)
        await self._warm_session_meta(str(user_id), row, message_ids=msg_ids)
        summaries = await self._fetch_session_summaries([session_id])
        count, preview = summaries.get(session_id, (0, None))
        return self._row_to_session_summary_public(row, count, preview)

    async def delete_session(self, user_id: uuid.UUID, session_id: str) -> bool:
        row = await self.get_session_for_user(user_id, session_id)
        if row is None:
            return False
        await self.db.execute(
            delete(AgentMessage).where(
                AgentMessage.session_id == session_id,
                AgentMessage.user_id == user_id,
            ),
        )
        row.deleted_at = datetime.now(timezone.utc)
        row.updated_at = datetime.now(timezone.utc)
        await self.db.flush()
        await self.cache.invalidate_session(str(user_id), session_id)
        return True

    async def prepare_turn(
        self,
        user_id: uuid.UUID,
        req: ChatStreamRequest,
        *,
        user_message_id: str,
        assistant_message_id: str,
        provider_model: str | None,
    ) -> tuple[list[ChatMessageInput], AgentSession]:
        row = await self.get_session_for_user(user_id, req.session_id)
        now = datetime.now(timezone.utc)
        title = _build_session_title(req.user_content)

        if row is None:
            foreign = await self.db.execute(
                select(AgentSession.id).where(
                    AgentSession.id == req.session_id,
                    AgentSession.user_id != user_id,
                ),
            )
            if foreign.scalar_one_or_none() is not None:
                raise PermissionError("session_forbidden")

            row = AgentSession(
                id=req.session_id,
                user_id=user_id,
                title=title,
                provider_id=req.provider_id,
                model=provider_model,
                context_summary=req.context.summary if req.context else None,
                summary_up_to_message_id=(
                    req.context.summary_up_to_message_id if req.context else None
                ),
                created_at=now,
                updated_at=now,
            )
            self.db.add(row)
        else:
            if row.title == "新对话" or req.truncate_from_message_id:
                row.title = title
            row.provider_id = req.provider_id
            if provider_model:
                row.model = provider_model
            row.updated_at = now

        self._sync_session_from_client_context(row, req.client_context)
        await self.db.flush()

        msg_mode = normalize_message_interaction_mode(
            getattr(req.client_context, "agent_mode", None) if req.client_context else None,
        )

        if req.truncate_from_message_id:
            await self._truncate_after_message(
                user_id,
                req.session_id,
                req.truncate_from_message_id,
            )
            await self._upsert_user_message(
                user_id=user_id,
                session_id=req.session_id,
                message_id=user_message_id,
                content=req.user_content,
                provider_id=req.provider_id,
                model=provider_model,
                replace_at_id=req.truncate_from_message_id,
                interaction_mode=msg_mode,
            )
        else:
            await self._append_user_message(
                user_id=user_id,
                session_id=req.session_id,
                message_id=user_message_id,
                content=req.user_content,
                provider_id=req.provider_id,
                model=provider_model,
                interaction_mode=msg_mode,
            )

        await self._append_assistant_placeholder(
            user_id=user_id,
            session_id=req.session_id,
            message_id=assistant_message_id,
            provider_id=req.provider_id,
            model=provider_model,
            interaction_mode=msg_mode,
        )

        chat_inputs = await self.build_chat_message_inputs(req.session_id)
        await self._rebuild_cache(user_id, req.session_id)
        return chat_inputs, row

    async def build_chat_message_inputs(
        self,
        session_id: str,
        *,
        up_to_message_id: str | None = None,
        exclude_message_ids: set[str] | None = None,
    ) -> list[ChatMessageInput]:
        result = await self.db.execute(
            select(AgentMessage)
            .where(
                AgentMessage.session_id == session_id,
                AgentMessage.status.in_(("done", "streaming", "stopped", "error")),
            )
            .order_by(AgentMessage.sort_index.asc()),
        )
        rows = list(result.scalars().all())
        max_sort: int | None = None
        if up_to_message_id:
            for row in rows:
                if row.id == up_to_message_id:
                    max_sort = row.sort_index
                    break
            if max_sort is not None:
                rows = [r for r in rows if r.sort_index <= max_sort]

        exclude = exclude_message_ids or set()
        inputs: list[ChatMessageInput] = []
        for row in rows:
            if row.id in exclude:
                continue
            if row.role not in ("user", "assistant"):
                continue
            text = blocks_to_text(row.blocks_json)
            if not text.strip():
                text = blocks_to_preview(row.blocks_json, max_len=400)
            if not text.strip() and row.status == "streaming":
                continue
            if not text.strip():
                continue
            parsed_mode: str | None = None
            if row.interaction_mode in ("chat", "annotation"):
                parsed_mode = row.interaction_mode
            inputs.append(
                ChatMessageInput(
                    role=row.role,  # type: ignore[arg-type]
                    content=text,
                    message_id=row.id,
                    interaction_mode=parsed_mode,  # type: ignore[arg-type]
                ),
            )
        return inputs

    async def apply_stream_event(
        self,
        user_id: uuid.UUID,
        session_id: str,
        message_id: str,
        event: StreamEventPayload,
    ) -> None:
        if event.type not in (
            "text_delta",
            "reasoning_delta",
            "tool_start",
            "tool_result",
            "annotation_progress",
            "annotation_proposal",
        ):
            return

        uid = str(user_id)
        cached = await self.cache.get_message(uid, message_id)
        if cached:
            blocks = apply_stream_event_to_blocks(cached.get("blocks", []), event)
            cached["blocks"] = blocks
            cached["updated_at_ms"] = int(datetime.now(timezone.utc).timestamp() * 1000)
            await self.cache.set_message(uid, session_id, cached)
            return

        result = await self.db.execute(
            select(AgentMessage).where(AgentMessage.id == message_id),
        )
        row = result.scalar_one_or_none()
        if row is None:
            return
        row.blocks_json = apply_stream_event_to_blocks(row.blocks_json, event)
        row.updated_at = datetime.now(timezone.utc)
        await self.db.flush()
        await self.cache.set_message(uid, session_id, self._row_to_message_cache(row))

    async def finalize_assistant_message(
        self,
        user_id: uuid.UUID,
        session_id: str,
        message_id: str,
        *,
        status: str,
        error: str | None = None,
    ) -> None:
        result = await self.db.execute(
            select(AgentMessage).where(AgentMessage.id == message_id),
        )
        row = result.scalar_one_or_none()
        uid = str(user_id)
        cached = await self.cache.get_message(uid, message_id)

        if row is not None:
            if cached and cached.get("blocks"):
                row.blocks_json = collapse_assistant_blocks(cached["blocks"])
            else:
                row.blocks_json = collapse_assistant_blocks(row.blocks_json)
            row.status = status
            row.error = error
            row.updated_at = datetime.now(timezone.utc)
            await self.db.flush()

        session_row = await self.get_session_for_user(user_id, session_id)
        if session_row:
            session_row.updated_at = datetime.now(timezone.utc)
            await self.db.flush()
            await self._rebuild_cache(user_id, session_id)

    async def try_advance_event_seq(
        self,
        user_id: uuid.UUID,
        client_job_id: str,
        seq: int | None,
    ) -> bool:
        """Return False if seq is duplicate (idempotent retry)."""
        if seq is None:
            return True
        key = f"agent:u:{user_id}:ann-run:{client_job_id}:seq"
        last_raw = await self.cache.redis.get(key)
        if last_raw is not None:
            try:
                last = int(last_raw)
            except ValueError:
                last = -1
            if seq <= last:
                return False
        ttl = self.settings.agent_job_cancel_ttl_seconds
        await self.cache.redis.set(key, str(seq), ex=ttl)
        return True

    async def apply_stream_events_batch(
        self,
        user_id: uuid.UUID,
        session_id: str,
        message_id: str,
        events: list[StreamEventPayload],
    ) -> None:
        for event in events:
            await self.apply_stream_event(user_id, session_id, message_id, event)

    async def finalize_annotation_turn(
        self,
        user_id: uuid.UUID,
        session_id: str,
        message_id: str,
        *,
        status: str,
        error: str | None = None,
        user_content: str | None = None,
    ) -> None:
        await self.finalize_assistant_message(
            user_id,
            session_id,
            message_id,
            status=status,
            error=error,
        )
        session_row = await self.get_session_for_user(user_id, session_id)
        if session_row is None:
            return
        result = await self.db.execute(
            select(AgentMessage).where(AgentMessage.id == message_id),
        )
        msg_row = result.scalar_one_or_none()
        if msg_row is not None:
            ann_title = _annotation_title_from_blocks(msg_row.blocks_json)
            if ann_title:
                session_row.title = ann_title
            elif user_content and session_row.title in ("新对话", ""):
                session_row.title = _build_session_title(user_content)
        session_row.updated_at = datetime.now(timezone.utc)
        await self.db.flush()
        await self._rebuild_cache(user_id, session_id)

    async def patch_message_block(
        self,
        user_id: uuid.UUID,
        session_id: str,
        message_id: str,
        *,
        block_type: str | None = None,
        block_index: int | None = None,
        patch: dict[str, Any],
    ) -> bool:
        session_row = await self.get_session_for_user(user_id, session_id)
        if session_row is None:
            return False
        result = await self.db.execute(
            select(AgentMessage).where(
                AgentMessage.id == message_id,
                AgentMessage.session_id == session_id,
                AgentMessage.user_id == user_id,
            ),
        )
        row = result.scalar_one_or_none()
        if row is None:
            return False

        blocks = [dict(b) for b in (row.blocks_json or [])]
        target_idx = block_index
        if target_idx is None and block_type:
            target_idx = next(
                (i for i, b in enumerate(blocks) if b.get("type") == block_type),
                -1,
            )
        if target_idx is None or target_idx < 0 or target_idx >= len(blocks):
            return False

        blocks[target_idx] = {**blocks[target_idx], **patch}
        row.blocks_json = blocks
        row.updated_at = datetime.now(timezone.utc)
        await self.db.flush()
        uid = str(user_id)
        await self.cache.set_message(uid, session_id, self._row_to_message_cache(row))
        session_row.updated_at = datetime.now(timezone.utc)
        await self.db.flush()
        await self._rebuild_cache(user_id, session_id)
        return True

    async def update_session_summary(
        self,
        user_id: uuid.UUID,
        session_id: str,
        *,
        summary: str,
        summary_up_to_message_id: str,
        token_estimate: int | None,
    ) -> None:
        row = await self.get_session_for_user(user_id, session_id)
        if row is None:
            return
        row.context_summary = summary
        row.summary_up_to_message_id = summary_up_to_message_id
        if token_estimate is not None:
            row.last_context_token_estimate = token_estimate
        row.updated_at = datetime.now(timezone.utc)
        await self.db.flush()
        await self._rebuild_cache(user_id, session_id)

    async def sync_context_from_request(
        self,
        user_id: uuid.UUID,
        req: ChatStreamRequest,
    ) -> None:
        if req.context is None:
            return
        row = await self.get_session_for_user(user_id, req.session_id)
        if row is None:
            return
        if req.context.summary is not None:
            row.context_summary = req.context.summary
        if req.context.summary_up_to_message_id is not None:
            row.summary_up_to_message_id = req.context.summary_up_to_message_id
        await self.db.flush()

    # --- internal helpers ---

    @staticmethod
    def _sync_session_from_client_context(row: AgentSession, client_context: Any | None) -> None:
        if client_context is None:
            return
        pid = getattr(client_context, "active_annotation_project_id", None)
        if pid and not row.annotation_project_id:
            row.annotation_project_id = str(pid)
        mode = getattr(client_context, "agent_mode", None)
        if mode in ("annotation", "annotate"):
            row.interaction_mode = "annotation"
        elif mode in ("chat", "ask", None):
            if mode == "chat":
                row.interaction_mode = "chat"

    async def _message_ids_from_db(self, session_id: str) -> list[str]:
        result = await self.db.execute(
            select(AgentMessage.id)
            .where(AgentMessage.session_id == session_id)
            .order_by(AgentMessage.sort_index.asc()),
        )
        return list(result.scalars().all())

    async def _next_sort_index(self, session_id: str) -> int:
        result = await self.db.execute(
            select(func.coalesce(func.max(AgentMessage.sort_index), -1)).where(
                AgentMessage.session_id == session_id,
            ),
        )
        current = result.scalar_one()
        return int(current) + 1

    async def _truncate_after_message(
        self,
        user_id: uuid.UUID,
        session_id: str,
        message_id: str,
    ) -> None:
        result = await self.db.execute(
            select(AgentMessage).where(
                AgentMessage.id == message_id,
                AgentMessage.session_id == session_id,
            ),
        )
        pivot = result.scalar_one_or_none()
        if pivot is None:
            return

        to_delete = await self.db.execute(
            select(AgentMessage.id).where(
                AgentMessage.session_id == session_id,
                AgentMessage.sort_index > pivot.sort_index,
            ),
        )
        delete_ids = list(to_delete.scalars().all())
        if delete_ids:
            await self.db.execute(
                delete(AgentMessage).where(AgentMessage.id.in_(delete_ids)),
            )
            await self.db.flush()

    async def _upsert_user_message(
        self,
        *,
        user_id: uuid.UUID,
        session_id: str,
        message_id: str,
        content: str,
        provider_id: str,
        model: str | None,
        replace_at_id: str,
        interaction_mode: str | None = None,
    ) -> None:
        result = await self.db.execute(
            select(AgentMessage).where(AgentMessage.id == replace_at_id),
        )
        row = result.scalar_one_or_none()
        blocks = [{"type": "text", "content": content}]
        now = datetime.now(timezone.utc)
        if row is None:
            sort_index = await self._next_sort_index(session_id)
            self.db.add(
                AgentMessage(
                    id=message_id,
                    session_id=session_id,
                    user_id=user_id,
                    role="user",
                    sort_index=sort_index,
                    blocks_json=blocks,
                    status="done",
                    interaction_mode=interaction_mode,
                    provider_id=provider_id,
                    model=model,
                    created_at=now,
                    updated_at=now,
                ),
            )
        else:
            row.blocks_json = blocks
            row.status = "done"
            row.provider_id = provider_id
            row.model = model
            row.interaction_mode = interaction_mode
            row.updated_at = now
        await self.db.flush()

    async def _append_user_message(
        self,
        *,
        user_id: uuid.UUID,
        session_id: str,
        message_id: str,
        content: str,
        provider_id: str,
        model: str | None,
        interaction_mode: str | None = None,
    ) -> None:
        existing = await self.db.execute(
            select(AgentMessage).where(AgentMessage.id == message_id),
        )
        existing_row = existing.scalar_one_or_none()
        if existing_row is not None:
            if existing_row.role == "user":
                return
            logger.warning(
                "user_message_id_conflict session=%s message_id=%s role=%s",
                session_id,
                message_id,
                existing_row.role,
            )
            raise ValueError("user_message_id_conflict")
        now = datetime.now(timezone.utc)
        sort_index = await self._next_sort_index(session_id)
        self.db.add(
            AgentMessage(
                id=message_id,
                session_id=session_id,
                user_id=user_id,
                role="user",
                sort_index=sort_index,
                blocks_json=[{"type": "text", "content": content}],
                status="done",
                interaction_mode=interaction_mode,
                provider_id=provider_id,
                model=model,
                created_at=now,
                updated_at=now,
            ),
        )
        await self.db.flush()

    async def _append_assistant_placeholder(
        self,
        *,
        user_id: uuid.UUID,
        session_id: str,
        message_id: str,
        provider_id: str,
        model: str | None,
        interaction_mode: str | None = None,
    ) -> None:
        existing = await self.db.execute(
            select(AgentMessage.id).where(AgentMessage.id == message_id),
        )
        if existing.scalar_one_or_none() is not None:
            await self.db.execute(
                update(AgentMessage)
                .where(AgentMessage.id == message_id)
                .values(
                    blocks_json=[],
                    status="streaming",
                    error=None,
                    interaction_mode=interaction_mode,
                    updated_at=datetime.now(timezone.utc),
                ),
            )
            await self.db.flush()
            return

        now = datetime.now(timezone.utc)
        sort_index = await self._next_sort_index(session_id)
        self.db.add(
            AgentMessage(
                id=message_id,
                session_id=session_id,
                user_id=user_id,
                role="assistant",
                sort_index=sort_index,
                blocks_json=[],
                status="streaming",
                interaction_mode=interaction_mode,
                provider_id=provider_id,
                model=model,
                created_at=now,
                updated_at=now,
            ),
        )
        await self.db.flush()

    async def _load_messages(
        self,
        session_id: str,
        user_id: uuid.UUID,
    ) -> list[AgentMessagePublic]:
        cached = await self.cache.get_messages_for_session(str(user_id), session_id)
        if cached is not None:
            return [self._cache_dict_to_message_public(item) for item in cached]

        result = await self.db.execute(
            select(AgentMessage)
            .where(AgentMessage.session_id == session_id)
            .order_by(AgentMessage.sort_index.asc()),
        )
        rows = result.scalars().all()
        if rows:
            await self._rebuild_cache(user_id, session_id)
        return [self._row_to_message_public(row) for row in rows]

    async def _rebuild_cache(self, user_id: uuid.UUID, session_id: str) -> None:
        uid = str(user_id)
        result = await self.db.execute(
            select(AgentSession).where(AgentSession.id == session_id),
        )
        session_row = result.scalar_one_or_none()
        if session_row is None:
            return

        msg_result = await self.db.execute(
            select(AgentMessage)
            .where(AgentMessage.session_id == session_id)
            .order_by(AgentMessage.sort_index.asc()),
        )
        messages = msg_result.scalars().all()
        msg_ids = [m.id for m in messages]
        await self._warm_session_meta(uid, session_row, message_ids=msg_ids)
        await self.cache.set_message_ids(uid, session_id, msg_ids)
        for message in messages:
            await self.cache.set_message(uid, session_id, self._row_to_message_cache(message))

    async def _warm_session_meta(
        self,
        user_id: str,
        row: AgentSession,
        *,
        message_ids: list[str] | None = None,
    ) -> None:
        if message_ids is None:
            message_ids = await self._message_ids_from_db(row.id)
        meta = {
            "title": row.title,
            "provider_id": row.provider_id or "",
            "model": row.model or "",
            "context_summary": row.context_summary or "",
            "summary_up_to_message_id": row.summary_up_to_message_id or "",
            "last_context_token_estimate": row.last_context_token_estimate or 0,
            "created_at_ms": _dt_to_ms(row.created_at),
            "updated_at_ms": _dt_to_ms(row.updated_at),
        }
        await self.cache.set_session_meta(user_id, row.id, meta)
        await self.cache.set_message_ids(user_id, row.id, message_ids)

    def _row_to_session_public(self, row: AgentSession) -> AgentSessionPublic:
        return self._row_to_session_summary_public(row, 0, None)

    def _row_to_session_summary_public(
        self,
        row: AgentSession,
        message_count: int,
        last_message_preview: str | None,
    ) -> AgentSessionPublic:
        preview = last_message_preview or None
        if preview == "":
            preview = None
        return AgentSessionPublic(
            id=row.id,
            title=row.title,
            annotation_project_id=row.annotation_project_id,
            interaction_mode=row.interaction_mode,
            provider_id=row.provider_id or "",
            model=row.model or "",
            message_ids=[],
            message_count=message_count,
            last_message_preview=preview,
            context_summary=row.context_summary,
            summary_up_to_message_id=row.summary_up_to_message_id,
            last_context_token_estimate=row.last_context_token_estimate,
            created_at=_dt_to_ms(row.created_at),
            updated_at=_dt_to_ms(row.updated_at),
        )

    async def _count_messages(self, session_id: str) -> int:
        result = await self.db.execute(
            select(func.count())
            .select_from(AgentMessage)
            .where(AgentMessage.session_id == session_id),
        )
        return int(result.scalar_one() or 0)

    async def _fetch_session_summaries(
        self,
        session_ids: list[str],
    ) -> dict[str, tuple[int, str | None]]:
        if not session_ids:
            return {}

        count_result = await self.db.execute(
            select(AgentMessage.session_id, func.count())
            .where(AgentMessage.session_id.in_(session_ids))
            .group_by(AgentMessage.session_id),
        )
        counts = {sid: int(count) for sid, count in count_result.all()}

        msg_result = await self.db.execute(
            select(AgentMessage)
            .where(AgentMessage.session_id.in_(session_ids))
            .order_by(AgentMessage.session_id, AgentMessage.sort_index.desc()),
        )
        previews: dict[str, str | None] = {}
        for message in msg_result.scalars().all():
            if message.session_id in previews:
                continue
            previews[message.session_id] = _build_message_preview(message.blocks_json) or None

        out: dict[str, tuple[int, str | None]] = {}
        for sid in session_ids:
            out[sid] = (counts.get(sid, 0), previews.get(sid))
        return out

    async def _load_messages_page(
        self,
        session_id: str,
        *,
        user_id: uuid.UUID,
        limit: int,
        before_message_id: str | None,
    ) -> tuple[list[AgentMessagePublic], bool]:
        pivot_index: int | None = None
        if before_message_id:
            pivot_result = await self.db.execute(
                select(AgentMessage.sort_index).where(
                    AgentMessage.id == before_message_id,
                    AgentMessage.session_id == session_id,
                ),
            )
            pivot_index = pivot_result.scalar_one_or_none()
            if pivot_index is None:
                raise ValueError("message_not_found")

        stmt = select(AgentMessage).where(AgentMessage.session_id == session_id)
        if pivot_index is not None:
            stmt = stmt.where(AgentMessage.sort_index < pivot_index)

        result = await self.db.execute(
            stmt.order_by(AgentMessage.sort_index.desc()).limit(limit + 1),
        )
        rows = list(result.scalars().all())
        has_more_before = len(rows) > limit
        page_rows = list(reversed(rows[:limit]))

        return [self._row_to_message_public(row) for row in page_rows], has_more_before

    def _row_to_message_public(self, row: AgentMessage) -> AgentMessagePublic:
        return AgentMessagePublic(
            id=row.id,
            session_id=row.session_id,
            role=row.role,
            blocks=row.blocks_json,
            status=row.status,
            interaction_mode=row.interaction_mode,
            provider_id=row.provider_id or "",
            model=row.model or "",
            error=row.error,
            created_at=_dt_to_ms(row.created_at),
            updated_at=_dt_to_ms(row.updated_at),
        )

    def _cache_dict_to_message_public(self, data: dict[str, Any]) -> AgentMessagePublic:
        return AgentMessagePublic(
            id=str(data["id"]),
            session_id=str(data["session_id"]),
            role=str(data["role"]),
            blocks=data.get("blocks") or [],
            status=str(data.get("status") or "done"),
            interaction_mode=data.get("interaction_mode"),
            provider_id=str(data.get("provider_id") or ""),
            model=str(data.get("model") or ""),
            error=data.get("error"),
            created_at=int(data.get("created_at_ms") or 0),
            updated_at=int(data.get("updated_at_ms") or 0),
        )

    def _row_to_message_cache(self, row: AgentMessage) -> dict[str, Any]:
        return {
            "id": row.id,
            "session_id": row.session_id,
            "role": row.role,
            "blocks": row.blocks_json,
            "status": row.status,
            "interaction_mode": row.interaction_mode,
            "provider_id": row.provider_id or "",
            "model": row.model or "",
            "error": row.error,
            "created_at_ms": _dt_to_ms(row.created_at),
            "updated_at_ms": _dt_to_ms(row.updated_at),
            "sort_index": row.sort_index,
        }
