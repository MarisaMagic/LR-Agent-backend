import json
import logging
from collections.abc import AsyncIterator

from fastapi import APIRouter, HTTPException, status
from fastapi.responses import StreamingResponse
from langchain_openai import ChatOpenAI

from app.agent.annotation.quality_report_compose_service import (
    stream_quality_report_compose,
)
from app.agent.text_sanitize import sanitize_json_value
from app.agent.llm_factory import build_chat_model
from app.core.deps import CurrentUser, DbSession, RedisClient, SettingsDep
from app.schemas.annotation_quality import QualityReportComposeRequest
from app.services.agent_rate_limit import check_agent_stream_limit
from app.services.llm_provider_service import LlmProviderService
import uuid

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/agent/annotation-quality", tags=["annotation-quality"])


async def _llm_for_provider(
    db: DbSession,
    settings: SettingsDep,
    user_id: uuid.UUID,
    provider_id: str,
    *,
    api_key_direct: str = "",
    base_url_direct: str = "",
    model_direct: str = "",
):
    # 前端直传模式：Electron 本地 provider 配置，跳过 DB 查询
    if api_key_direct.strip() and base_url_direct.strip() and model_direct.strip():
        return ChatOpenAI(
            model=model_direct.strip(),
            api_key=api_key_direct.strip(),
            base_url=base_url_direct.strip().rstrip("/"),
            streaming=True,
            temperature=0.2,
            timeout=120,
        )

    svc = LlmProviderService(db, settings)
    try:
        provider_uuid = uuid.UUID(provider_id)
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail="invalid_provider_id"
        ) from exc

    row = await svc.get_for_user(provider_uuid, user_id)
    if row is None or not row.enabled:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="llm_provider_not_found"
        )

    try:
        svc.validate_provider_base_url(row)
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail="invalid_base_url"
        ) from exc

    api_key = svc.decrypt_api_key(row)
    return build_chat_model(row, api_key, streaming=True, temperature=0.2)


def _validate_compose_payload(payload: dict, settings: SettingsDep) -> None:
    payload_bytes = len(json.dumps(payload, ensure_ascii=False).encode("utf-8"))
    max_bytes = getattr(settings, "agent_analysis_snapshot_max_bytes", 512_000)
    if payload_bytes > max_bytes:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="compose_payload_too_large",
        )


async def _compose_sse(
    llm,
    *,
    compose_payload: dict,
    settings: SettingsDep,
) -> AsyncIterator[str]:
    try:
        async for event in stream_quality_report_compose(
            llm,
            compose_payload=compose_payload,
        ):
            yield f"data: {json.dumps(event.to_sse_dict(), ensure_ascii=False)}\n\n"
        yield f"data: {json.dumps({'type': 'done'}, ensure_ascii=False)}\n\n"
    except Exception:
        logger.exception("quality_report_compose_stream_error")
        err = {"type": "error", "message": "quality_report_compose_failed"}
        if settings.debug:
            err["detail"] = "quality_report_compose_failed"
        yield f"data: {json.dumps(err, ensure_ascii=False)}\n\n"


@router.post("/report/compose/stream", summary="流式撰写标注质量报告")
async def api_quality_report_compose_stream(
    body: QualityReportComposeRequest,
    current_user: CurrentUser,
    db: DbSession,
    redis: RedisClient,
    settings: SettingsDep,
) -> StreamingResponse:
    _validate_compose_payload(body.compose_payload, settings)
    await check_agent_stream_limit(redis, settings, current_user.id)

    llm = await _llm_for_provider(
        db,
        settings,
        current_user.id,
        body.provider_id,
        api_key_direct=body.api_key,
        base_url_direct=body.base_url,
        model_direct=body.model,
    )

    return StreamingResponse(
        _compose_sse(
            llm,
            compose_payload=sanitize_json_value(body.compose_payload),
            settings=settings,
        ),
        media_type="text/event-stream",
    )
