import uuid
from datetime import datetime, timezone

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.agent_message import AgentMessage
from app.models.agent_session import AgentSession
from app.schemas.agent import ChatStreamRequest


class AgentSessionService:
    def __init__(self, db: AsyncSession) -> None:
        self.db = db

    async def upsert_session(
        self,
        user_id: uuid.UUID,
        req: ChatStreamRequest,
        *,
        title: str | None = None,
        context_summary: str | None = None,
        summary_up_to_message_id: str | None = None,
        token_estimate: int | None = None,
    ) -> AgentSession:
        result = await self.db.execute(
            select(AgentSession).where(
                AgentSession.id == req.session_id,
                AgentSession.user_id == user_id,
            ),
        )
        row = result.scalar_one_or_none()
        now = datetime.now(timezone.utc)

        if row is None:
            row = AgentSession(
                id=req.session_id,
                user_id=user_id,
                title=title or "新对话",
                provider_id=req.provider_id,
                model=None,
                context_summary=context_summary,
                summary_up_to_message_id=summary_up_to_message_id,
                last_context_token_estimate=token_estimate,
                created_at=now,
                updated_at=now,
            )
            self.db.add(row)
        else:
            if title:
                row.title = title
            row.provider_id = req.provider_id
            if context_summary is not None:
                row.context_summary = context_summary
            if summary_up_to_message_id is not None:
                row.summary_up_to_message_id = summary_up_to_message_id
            if token_estimate is not None:
                row.last_context_token_estimate = token_estimate
            row.updated_at = now

        await self.db.flush()
        return row

    async def append_message(
        self,
        user_id: uuid.UUID,
        *,
        message_id: str,
        session_id: str,
        role: str,
        blocks_json: list,
        status: str,
        provider_id: str | None,
        model: str | None,
    ) -> None:
        result = await self.db.execute(
            select(AgentMessage).where(AgentMessage.id == message_id),
        )
        if result.scalar_one_or_none() is not None:
            return

        self.db.add(
            AgentMessage(
                id=message_id,
                session_id=session_id,
                user_id=user_id,
                role=role,
                blocks_json=blocks_json,
                status=status,
                provider_id=provider_id,
                model=model,
            ),
        )
        await self.db.flush()
