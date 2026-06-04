import json
import logging
import time
from collections.abc import AsyncIterator

from fastapi import APIRouter, HTTPException, Query, status
from fastapi.responses import StreamingResponse

from app.agent.orchestrator import ChatOrchestrator
from app.core.deps import CurrentUser, DbSession, RedisClient, SettingsDep
from app.schemas.agent import (
    AgentMessageBlockPatchRequest,
    AgentSessionCreateRequest,
    AgentSessionDetailResponse,
    AgentSessionListResponse,
    AgentSessionPatchRequest,
    AgentSessionPublic,
    AnnotationRunEventsRequest,
    AnnotationRunFinalizeRequest,
    AnnotationRunStartRequest,
    ChatCancelRequest,
    ChatStreamRequest,
)
from app.services.agent_chat_repository import AgentChatRepository
from app.services.agent_job_service import AgentJobService
from app.services.annotation_run_service import AnnotationRunService
from app.services.agent_rate_limit import (
    check_agent_session_write_limit,
    check_agent_stream_limit,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/agent", tags=["agent"])


def _new_session_id() -> str:
    suffix = format(time.time_ns() % 1_000_000_000, "x")
    return f"session-{suffix}"


async def _sse_stream(
    orchestrator: ChatOrchestrator,
    user: CurrentUser,
    body: ChatStreamRequest,
    *,
    settings: SettingsDep,
) -> AsyncIterator[str]:
    try:
        async for event in orchestrator.run(user, body):
            payload = event.to_sse_dict()
            yield f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"
    except Exception:
        logger.exception("sse stream failed user=%s session=%s", user.id, body.session_id)
        err = {"type": "error", "message": "stream_failed"}
        if settings.debug:
            err["detail"] = "stream_failed"
        yield f"data: {json.dumps(err, ensure_ascii=False)}\n\n"


@router.get("/sessions", response_model=AgentSessionListResponse)
async def list_agent_sessions(
    current_user: CurrentUser,
    db: DbSession,
    redis: RedisClient,
    settings: SettingsDep,
    limit: int = Query(default=None, ge=1),
    cursor: str | None = None,
    annotation_project_id: str | None = Query(default=None, max_length=64),
    workspace_only: bool = Query(default=False),
) -> AgentSessionListResponse:
    page_limit = limit or settings.agent_session_list_default_limit
    page_limit = min(page_limit, settings.agent_session_list_max_limit)
    repo = AgentChatRepository(db, redis, settings)
    try:
        sessions, next_cursor, has_more = await repo.list_sessions(
            current_user.id,
            limit=page_limit,
            cursor=cursor,
            annotation_project_id=annotation_project_id,
            workspace_only=workspace_only,
        )
    except ValueError as exc:
        if str(exc) == "invalid_cursor":
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="invalid_cursor") from exc
        raise
    return AgentSessionListResponse(
        sessions=sessions,
        next_cursor=next_cursor,
        has_more=has_more,
    )


@router.post("/sessions", response_model=AgentSessionPublic, status_code=status.HTTP_201_CREATED)
async def create_agent_session(
    body: AgentSessionCreateRequest,
    current_user: CurrentUser,
    db: DbSession,
    redis: RedisClient,
    settings: SettingsDep,
) -> AgentSessionPublic:
    await check_agent_session_write_limit(redis, settings, current_user.id)
    repo = AgentChatRepository(db, redis, settings)
    session_id = body.id or _new_session_id()
    existing = await repo.get_session_for_user(current_user.id, session_id)
    if existing is not None:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail="session_exists")
    return await repo.create_session(
        current_user.id,
        session_id=session_id,
        title=body.title,
        provider_id=body.provider_id,
        model=body.model,
        annotation_project_id=body.annotation_project_id,
        interaction_mode=body.interaction_mode,
    )


@router.get("/sessions/{session_id}", response_model=AgentSessionDetailResponse)
async def get_agent_session(
    session_id: str,
    current_user: CurrentUser,
    db: DbSession,
    redis: RedisClient,
    settings: SettingsDep,
    limit: int = Query(default=None, ge=1),
    before_message_id: str | None = None,
) -> AgentSessionDetailResponse:
    page_limit = limit or settings.agent_message_page_default_limit
    page_limit = min(page_limit, settings.agent_message_page_max_limit)
    repo = AgentChatRepository(db, redis, settings)
    try:
        detail = await repo.get_session_detail(
            current_user.id,
            session_id,
            limit=page_limit,
            before_message_id=before_message_id,
        )
    except ValueError as exc:
        if str(exc) == "message_not_found":
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="message_not_found",
            ) from exc
        raise
    if detail is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="session_not_found")
    return detail


@router.patch("/sessions/{session_id}", response_model=AgentSessionPublic)
async def patch_agent_session(
    session_id: str,
    body: AgentSessionPatchRequest,
    current_user: CurrentUser,
    db: DbSession,
    redis: RedisClient,
    settings: SettingsDep,
) -> AgentSessionPublic:
    await check_agent_session_write_limit(redis, settings, current_user.id)
    repo = AgentChatRepository(db, redis, settings)
    updated = await repo.update_session(
        current_user.id,
        session_id,
        title=body.title,
        provider_id=body.provider_id,
        model=body.model,
    )
    if updated is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="session_not_found")
    return updated


@router.delete("/sessions/{session_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_agent_session(
    session_id: str,
    current_user: CurrentUser,
    db: DbSession,
    redis: RedisClient,
    settings: SettingsDep,
) -> None:
    await check_agent_session_write_limit(redis, settings, current_user.id)
    repo = AgentChatRepository(db, redis, settings)
    deleted = await repo.delete_session(current_user.id, session_id)
    if not deleted:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="session_not_found")


@router.post("/chat/stream")
async def chat_stream(
    body: ChatStreamRequest,
    current_user: CurrentUser,
    db: DbSession,
    redis: RedisClient,
    settings: SettingsDep,
) -> StreamingResponse:
    await check_agent_stream_limit(redis, settings, current_user.id)
    jobs = AgentJobService(redis, settings)
    orchestrator = ChatOrchestrator(db, redis, jobs, settings)

    return StreamingResponse(
        _sse_stream(orchestrator, current_user, body, settings=settings),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@router.post("/chat/cancel")
async def chat_cancel(
    body: ChatCancelRequest,
    current_user: CurrentUser,
    redis: RedisClient,
    settings: SettingsDep,
) -> dict[str, str]:
    jobs = AgentJobService(redis, settings)
    cancelled = await jobs.mark_cancelled(str(current_user.id), body.client_job_id)
    if not cancelled:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="job_not_found")
    return {"status": "cancelled"}


def _annotation_run_http_error(exc: ValueError) -> HTTPException:
    detail = str(exc)
    code = status.HTTP_400_BAD_REQUEST
    if detail in ("session_not_found", "llm_provider_not_found"):
        code = status.HTTP_404_NOT_FOUND
    if detail == "session_forbidden":
        code = status.HTTP_403_FORBIDDEN
    return HTTPException(status_code=code, detail=detail)


@router.post("/annotation-run/start")
async def annotation_run_start(
    body: AnnotationRunStartRequest,
    current_user: CurrentUser,
    db: DbSession,
    redis: RedisClient,
    settings: SettingsDep,
) -> dict[str, str]:
    await check_agent_session_write_limit(redis, settings, current_user.id)
    svc = AnnotationRunService(db, redis, settings)
    try:
        return await svc.start_turn(current_user, body)
    except ValueError as exc:
        raise _annotation_run_http_error(exc) from exc


@router.post("/annotation-run/events")
async def annotation_run_events(
    body: AnnotationRunEventsRequest,
    current_user: CurrentUser,
    db: DbSession,
    redis: RedisClient,
    settings: SettingsDep,
) -> dict[str, bool]:
    svc = AnnotationRunService(db, redis, settings)
    try:
        return await svc.ingest_events(current_user, body)
    except ValueError as exc:
        raise _annotation_run_http_error(exc) from exc


@router.post("/annotation-run/finalize")
async def annotation_run_finalize(
    body: AnnotationRunFinalizeRequest,
    current_user: CurrentUser,
    db: DbSession,
    redis: RedisClient,
    settings: SettingsDep,
) -> dict[str, bool]:
    await check_agent_session_write_limit(redis, settings, current_user.id)
    svc = AnnotationRunService(db, redis, settings)
    try:
        return await svc.finalize_turn(current_user, body)
    except ValueError as exc:
        raise _annotation_run_http_error(exc) from exc


@router.patch("/messages/{message_id}/blocks")
async def patch_agent_message_block(
    message_id: str,
    body: AgentMessageBlockPatchRequest,
    current_user: CurrentUser,
    db: DbSession,
    redis: RedisClient,
    settings: SettingsDep,
    session_id: str = Query(..., min_length=1, max_length=64),
) -> dict[str, bool]:
    await check_agent_session_write_limit(redis, settings, current_user.id)
    repo = AgentChatRepository(db, redis, settings)
    ok = await repo.patch_message_block(
        current_user.id,
        session_id,
        message_id,
        block_type=body.block_type,
        block_index=body.block_index,
        patch=body.patch,
    )
    if not ok:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="message_or_block_not_found")
    return {"ok": True}
