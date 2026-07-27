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
from app.core.deps import CurrentUser, RedisClient, SettingsDep
from app.schemas.annotation_quality import QualityReportComposeRequest
from app.services.agent_rate_limit import check_agent_stream_limit

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/agent/annotation-quality", tags=["annotation-quality"])


def _require_direct_llm(
    *,
    api_key: str,
    base_url: str,
    model: str,
) -> ChatOpenAI:
    if not api_key.strip() or not base_url.strip() or not model.strip():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="provider_credentials_required",
        )
    return ChatOpenAI(
        model=model.strip(),
        api_key=api_key.strip(),
        base_url=base_url.strip().rstrip("/"),
        streaming=True,
        temperature=0.2,
        timeout=120,
    )


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
    redis: RedisClient,
    settings: SettingsDep,
) -> StreamingResponse:
    del current_user
    _validate_compose_payload(body.compose_payload, settings)
    await check_agent_stream_limit(redis, settings, current_user.id)

    llm = _require_direct_llm(
        api_key=body.api_key,
        base_url=body.base_url,
        model=body.model,
    )

    return StreamingResponse(
        _compose_sse(
            llm,
            compose_payload=sanitize_json_value(body.compose_payload),
            settings=settings,
        ),
        media_type="text/event-stream",
    )
