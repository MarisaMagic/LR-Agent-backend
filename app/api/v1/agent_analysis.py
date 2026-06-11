import json
import logging
import uuid
from collections.abc import AsyncIterator

from fastapi import APIRouter, HTTPException, status
from fastapi.responses import StreamingResponse
from openai import BadRequestError
from pydantic import BaseModel, Field

from app.agent.analysis_prepare_service import (
    AnalysisRepairContext,
    prepare_analysis_script,
)
from app.agent.text_sanitize import sanitize_json_value
from app.agent.analysis_summarize_service import stream_analysis_summary
from app.agent.conversation_context import load_conversation_transcript
from app.agent.llm_factory import build_chat_model
from app.core.deps import CurrentUser, DbSession, RedisClient, SettingsDep
from app.services.agent_chat_repository import AgentChatRepository
from app.services.agent_rate_limit import check_agent_analysis_limit, check_agent_stream_limit
from app.services.llm_provider_service import LlmProviderService

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/agent/analysis", tags=["agent-analysis"])


class AnalysisPrepareRequest(BaseModel):
    provider_id: str = Field(min_length=1, max_length=64)
    user_request: str = Field(min_length=1, max_length=20_000)
    data_snapshot: dict = Field(default_factory=dict)
    session_id: str | None = Field(default=None, max_length=64)
    repair_context: AnalysisRepairContext | None = None


class AnalysisSummarizeRequest(BaseModel):
    provider_id: str = Field(min_length=1, max_length=64)
    user_request: str = Field(min_length=1, max_length=20_000)
    session_id: str | None = Field(default=None, max_length=64)
    script: str = Field(min_length=1, max_length=32_000)
    explanation: str = ""
    stdout: str = Field(default="", max_length=32_000)


async def _llm_for_analysis_provider(
    db: DbSession,
    settings: SettingsDep,
    user_id: uuid.UUID,
    provider_id: str,
    *,
    streaming: bool = False,
    temperature: float = 0.0,
):
    snapshot_max = settings.agent_analysis_snapshot_max_bytes
    _ = snapshot_max  # callers validate snapshot size separately

    svc = LlmProviderService(db, settings)
    try:
        provider_uuid = uuid.UUID(provider_id)
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="invalid_provider_id") from exc

    row = await svc.get_for_user(provider_uuid, user_id)
    if row is None or not row.enabled:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="llm_provider_not_found")

    try:
        svc.validate_provider_base_url(row)
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="invalid_base_url") from exc

    api_key = svc.decrypt_api_key(row)
    return build_chat_model(row, api_key, streaming=streaming, temperature=temperature)


def _validate_snapshot_size(data_snapshot: dict, settings: SettingsDep) -> None:
    snapshot_bytes = len(json.dumps(data_snapshot, ensure_ascii=False).encode("utf-8"))
    if snapshot_bytes > settings.agent_analysis_snapshot_max_bytes:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="data_snapshot_too_large",
        )


@router.post("/prepare", summary="生成数据分析 Python 脚本")
async def api_analysis_prepare(
    body: AnalysisPrepareRequest,
    current_user: CurrentUser,
    db: DbSession,
    redis: RedisClient,
    settings: SettingsDep,
):
    if not settings.agent_analysis_enabled:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="agent_analysis_disabled",
        )

    _validate_snapshot_size(body.data_snapshot, settings)
    await check_agent_analysis_limit(redis, settings, current_user.id)

    llm = await _llm_for_analysis_provider(
        db, settings, current_user.id, body.provider_id, streaming=False, temperature=0.0
    )

    repo = AgentChatRepository(db, redis, settings)
    conversation_transcript = await load_conversation_transcript(
        repo,
        body.session_id,
        user_id=current_user.id,
        user_request=body.user_request,
        max_turns_in_window=settings.agent_default_max_turns_in_window,
    )

    try:
        result = await prepare_analysis_script(
            llm,
            user_request=body.user_request,
            data_snapshot=sanitize_json_value(body.data_snapshot),
            conversation_transcript=conversation_transcript,
            repair_context=body.repair_context,
        )
        return {"data": result.model_dump()}
    except BadRequestError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("analysis_prepare_error")
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=str(exc)[:500],
        ) from exc


async def _analysis_summarize_sse(
    llm,
    *,
    user_request: str,
    explanation: str,
    script: str,
    stdout: str,
    conversation_transcript: str,
    settings: SettingsDep,
) -> AsyncIterator[str]:
    try:
        async for event in stream_analysis_summary(
            llm,
            user_request=user_request,
            explanation=explanation,
            script=script,
            stdout=stdout,
            conversation_transcript=conversation_transcript,
        ):
            yield f"data: {json.dumps(event.to_sse_dict(), ensure_ascii=False)}\n\n"
        yield f"data: {json.dumps({'type': 'done'}, ensure_ascii=False)}\n\n"
    except Exception:
        logger.exception("analysis_summarize_stream_error")
        err = {"type": "error", "message": "summarize_stream_failed"}
        if settings.debug:
            err["detail"] = "summarize_stream_failed"
        yield f"data: {json.dumps(err, ensure_ascii=False)}\n\n"


@router.post("/summarize/stream", summary="流式解读数据分析 stdout")
async def api_analysis_summarize_stream(
    body: AnalysisSummarizeRequest,
    current_user: CurrentUser,
    db: DbSession,
    redis: RedisClient,
    settings: SettingsDep,
) -> StreamingResponse:
    if not settings.agent_analysis_enabled:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="agent_analysis_disabled",
        )

    await check_agent_stream_limit(redis, settings, current_user.id)

    llm = await _llm_for_analysis_provider(
        db, settings, current_user.id, body.provider_id, streaming=True, temperature=0.2
    )

    repo = AgentChatRepository(db, redis, settings)
    conversation_transcript = await load_conversation_transcript(
        repo,
        body.session_id,
        user_id=current_user.id,
        user_request=body.user_request,
        max_turns_in_window=settings.agent_default_max_turns_in_window,
    )

    return StreamingResponse(
        _analysis_summarize_sse(
            llm,
            user_request=body.user_request,
            explanation=body.explanation,
            script=body.script,
            stdout=body.stdout,
            conversation_transcript=conversation_transcript,
            settings=settings,
        ),
        media_type="text/event-stream",
    )
