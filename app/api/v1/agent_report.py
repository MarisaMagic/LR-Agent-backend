import logging
import uuid

from fastapi import APIRouter, HTTPException, status
from openai import BadRequestError
from pydantic import BaseModel, Field

from app.agent.conversation_context import load_conversation_transcript
from app.agent.report_prepare_service import ReportKind, prepare_report_markdown
from app.agent.llm_factory import build_chat_model
from app.core.deps import CurrentUser, DbSession, RedisClient, SettingsDep
from app.services.agent_chat_repository import AgentChatRepository
from app.services.agent_rate_limit import check_agent_analysis_limit
from app.services.llm_provider_service import LlmProviderService

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/agent/report", tags=["agent-report"])


class ReportPrepareRequest(BaseModel):
    provider_id: str = Field(min_length=1, max_length=64)
    user_request: str = Field(min_length=1, max_length=20_000)
    data_snapshot: dict = Field(default_factory=dict)
    report_kind: ReportKind = "report"
    session_id: str | None = Field(default=None, max_length=64)


@router.post("/prepare", summary="生成 Markdown 报告/文档提案")
async def api_report_prepare(
    body: ReportPrepareRequest,
    current_user: CurrentUser,
    db: DbSession,
    redis: RedisClient,
    settings: SettingsDep,
):
    if not settings.agent_document_write_enabled:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="agent_document_write_disabled",
        )

    import json

    snapshot_bytes = len(json.dumps(body.data_snapshot, ensure_ascii=False).encode("utf-8"))
    if snapshot_bytes > settings.agent_analysis_snapshot_max_bytes:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="data_snapshot_too_large",
        )

    await check_agent_analysis_limit(redis, settings, current_user.id)

    svc = LlmProviderService(db, settings)
    try:
        provider_uuid = uuid.UUID(body.provider_id)
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="invalid_provider_id") from exc

    row = await svc.get_for_user(provider_uuid, current_user.id)
    if row is None or not row.enabled:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="llm_provider_not_found")

    try:
        svc.validate_provider_base_url(row)
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="invalid_base_url") from exc

    api_key = svc.decrypt_api_key(row)
    llm = build_chat_model(row, api_key, streaming=False, temperature=0.2)

    repo = AgentChatRepository(db, redis, settings)
    conversation_transcript = await load_conversation_transcript(
        repo,
        body.session_id,
        user_id=current_user.id,
        user_request=body.user_request,
        max_turns_in_window=settings.agent_default_max_turns_in_window,
    )

    try:
        result = await prepare_report_markdown(
            llm,
            user_request=body.user_request,
            data_snapshot=body.data_snapshot,
            report_kind=body.report_kind,
            conversation_transcript=conversation_transcript,
        )
        return {"data": result.model_dump()}
    except BadRequestError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("report_prepare_error")
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=str(exc)[:500],
        ) from exc
