import logging
import uuid

from fastapi import APIRouter, HTTPException, status
from openai import BadRequestError

from app.agent.annotation import (
    judge_detection_labels,
    prepare_batch_annotation,
    heuristic_map_boxes,
    map_detection_boxes_to_labels_unified,
)
from app.agent.annotation.mutation_prepare_service import prepare_mutation_annotation
from app.agent.annotation.debug_log import log_annotation_agent
from app.agent.annotation.image_bytes_loader import load_image_bytes
from app.agent.annotation.schemas import AnnotationScopePayload
from app.agent.llm_factory import build_chat_model
from app.agent.conversation_context import load_conversation_transcript
from app.core.deps import CurrentUser, DbSession, RedisClient, SettingsDep
from app.services.agent_chat_repository import AgentChatRepository
from app.schemas.annotation_agent import (
    BatchPrepareRequest,
    HeuristicMapRequest,
    JudgeDetectionLabelsRequest,
    MapDetectionBoxesRequest,
    MutationPrepareRequest,
)
from app.services.llm_provider_service import LlmProviderService

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/agent/annotation", tags=["agent-annotation"])


def _http_from_llm_error(exc: Exception) -> HTTPException:
    msg = str(exc)
    if isinstance(exc, BadRequestError):
        return HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=msg or "llm_bad_request",
        )
    if "response_format" in msg.lower() or "json_schema" in msg.lower():
        return HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="当前模型不支持结构化输出，请使用普通对话模型或更换提供商",
        )
    logger.exception("annotation_llm_error")
    return HTTPException(
        status_code=status.HTTP_502_BAD_GATEWAY,
        detail=msg[:500] if msg else "llm_invoke_failed",
    )


async def _llm_for_provider(
    db: DbSession,
    settings: SettingsDep,
    user_id: uuid.UUID,
    provider_id: str,
    *,
    temperature: float = 0.0,
    api_key_direct: str = "",
    base_url_direct: str = "",
    model_direct: str = "",
):
    # 前端直传模式：跳过 DB 查询
    if api_key_direct.strip() and base_url_direct.strip() and model_direct.strip():
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(
            model=model_direct.strip(),
            api_key=api_key_direct.strip(),
            base_url=base_url_direct.strip().rstrip("/"),
            streaming=False,
            temperature=temperature,
            timeout=120,
        )

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
    return build_chat_model(row, api_key, streaming=False, temperature=temperature)


@router.post("/map-heuristic", summary="启发式检测框映射（无 LLM）")
async def api_map_heuristic(
    body: HeuristicMapRequest,
    current_user: CurrentUser,
):
    del current_user
    mappings = heuristic_map_boxes(
        body.boxes,
        body.label_candidates,
        ocr_text=body.ocr_text,
    )
    return {"data": {"mappings": mappings, "method": "heuristic"}}


@router.post("/mutation-prepare", summary="标注变更准备（改标签/删框，单次 LLM）")
async def api_mutation_prepare(
    body: MutationPrepareRequest,
    current_user: CurrentUser,
    db: DbSession,
    redis: RedisClient,
    settings: SettingsDep,
):
    if not settings.agent_mutation_enabled:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="agent_mutation_disabled",
        )
    try:
        llm = await _llm_for_provider(
            db,
            settings,
            current_user.id,
            body.provider_id,
            temperature=settings.annotation_prepare_temperature,
            api_key_direct=body.api_key,
            base_url_direct=body.base_url,
            model_direct=body.model,
        )
        label_names = None
        if body.project is not None:
            label_names = [
                str(l.get("name") or "")
                for l in (body.project.labels or [])
                if str(l.get("name") or "").strip()
            ]
        repo = AgentChatRepository(db, redis, settings)
        conversation_transcript = await load_conversation_transcript(
            repo,
            body.session_id,
            user_id=current_user.id,
            user_request=body.user_request,
            max_turns_in_window=settings.agent_default_max_turns_in_window,
        )

        result = await prepare_mutation_annotation(
            llm,
            user_request=body.user_request,
            current_relative_path=(body.current_relative_path or "").strip(),
            candidates=[c.model_dump() for c in body.candidates],
            label_names=label_names or [],
            selected_annotation_ids=body.selected_annotation_ids or None,
            conversation_transcript=conversation_transcript,
        )
        return {
            "data": {
                **result.model_dump(),
                "resolved_user_request": body.user_request,
            }
        }
    except HTTPException:
        raise
    except Exception as exc:
        raise _http_from_llm_error(exc) from exc


@router.post("/batch-prepare", summary="批量准备（范围+计划，单次 LLM）")
async def api_batch_prepare(
    body: BatchPrepareRequest,
    current_user: CurrentUser,
    db: DbSession,
    redis: RedisClient,
    settings: SettingsDep,
):
    try:
        llm = await _llm_for_provider(
            db,
            settings,
            current_user.id,
            body.provider_id,
            temperature=settings.annotation_prepare_temperature,
            api_key_direct=body.api_key,
            base_url_direct=body.base_url,
            model_direct=body.model,
        )
        is_direct = bool(body.api_key.strip() and body.base_url.strip() and body.model.strip())
        if is_direct:
            provider_is_vision = body.supports_vision
        else:
            svc = LlmProviderService(db, settings)
            provider_uuid = uuid.UUID(body.provider_id)
            row = await svc.get_for_user(provider_uuid, current_user.id)
            if row is None:
                raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="llm_provider_not_found")
            provider_is_vision = await svc.ensure_vision_probed(row)

        label_names = None
        project_name = None
        if body.project is not None:
            project_name = body.project.name or None
            label_names = [
                str(l.get("name") or "")
                for l in (body.project.labels or [])
                if str(l.get("name") or "").strip()
            ]
        user_request = body.user_request
        conversation_transcript = ""
        if not body.preselected_paths:
            repo = AgentChatRepository(db, redis, settings)
            conversation_transcript = await load_conversation_transcript(
                repo,
                body.session_id,
                user_id=current_user.id,
                user_request=body.user_request,
                max_turns_in_window=settings.agent_default_max_turns_in_window,
            )

        current_rel = (body.current_relative_path or "").strip()
        candidates = [c.model_dump() for c in body.candidates]
        result = await prepare_batch_annotation(
            llm,
            user_request=user_request,
            current_relative_path=current_rel,
            candidates=candidates,
            label_candidates=body.label_candidates,
            detection_models=body.detection_models,
            default_conf=body.default_conf_threshold,
            default_iou=body.default_iou_threshold,
            provider_is_vision=provider_is_vision,
            project_name=project_name,
            label_names=label_names,
            conversation_transcript=conversation_transcript,
            preselected_paths=body.preselected_paths or None,
        )
        payload = {
            "selected_paths": result.selected_paths,
            "scope_reason": result.scope_reason,
            "resolved_user_request": user_request,
            "judge_config": {
                "enabled": settings.annotation_judge_enabled and provider_is_vision,
                "max_retries": settings.annotation_judge_max_retries,
            },
            **result.plan.model_dump(),
        }
        return {"data": payload}
    except HTTPException:
        raise
    except Exception as exc:
        raise _http_from_llm_error(exc) from exc


@router.post("/map-detection-boxes", summary="Fusion 统一检测框映射（启发式或逐框视觉）")
async def api_map_detection_boxes(
    body: MapDetectionBoxesRequest,
    current_user: CurrentUser,
    db: DbSession,
    settings: SettingsDep,
):
    try:
        is_direct = bool(body.api_key.strip() and body.base_url.strip() and body.model.strip())
        if is_direct:
            provider_is_vision = body.supports_vision
            provider_name = body.model
            provider_model = body.model
        else:
            svc = LlmProviderService(db, settings)
            row = await svc.get_for_user(uuid.UUID(body.provider_id), current_user.id)
            if row is None:
                raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="llm_provider_not_found")
            provider_is_vision = await svc.ensure_vision_probed(row)
            provider_name = row.name
            provider_model = row.model
            vision_probe_detail = row.vision_probe_detail
        use_vision_requested = bool(body.use_vision)
        use_vision = use_vision_requested and provider_is_vision
        llm = None
        if use_vision:
            llm = await _llm_for_provider(
                db,
                settings,
                current_user.id,
                body.provider_id,
                temperature=settings.annotation_llm_temperature,
                api_key_direct=body.api_key,
                base_url_direct=body.base_url,
                model_direct=body.model,
            )
        log_annotation_agent(
            "map-api",
            "map-detection-boxes 请求",
            provider_id=body.provider_id,
            provider_name=provider_name,
            provider_model=provider_model,
            provider_is_vision=provider_is_vision,
            vision_probe_detail=vision_probe_detail if not is_direct else "direct",
            use_vision_requested=use_vision_requested,
            use_vision_effective=use_vision,
            box_count=len(body.boxes),
            has_image_bytes=bool(
                load_image_bytes(
                    image_absolute_path=body.image_absolute_path,
                    image_base64=body.image_base64,
                )[0]
            ),
            image_absolute_path=bool(body.image_absolute_path.strip()),
            label_names=[str(c.get("name") or "") for c in (body.label_candidates or [])[:20]],
        )
        scope = AnnotationScopePayload.model_validate(body.annotation_scope or {})
        result = await map_detection_boxes_to_labels_unified(
            llm,
            user_request=body.user_request,
            intent_summary=body.intent_summary,
            label_candidates=body.label_candidates,
            boxes=body.boxes,
            use_vision=use_vision,
            ocr_text=body.ocr_text,
            scope=scope,
            label_strategy=body.label_strategy,
            single_label_id=body.single_label_id,
            image_absolute_path=body.image_absolute_path,
            image_base64=body.image_base64,
            mime_type=body.mime_type,
            judge_feedback=body.judge_feedback,
            previous_mappings=body.previous_mappings,
            attempt=body.attempt,
        )
        log_annotation_agent(
            "map-api-result",
            "map-detection-boxes 响应",
            ok=result.get("ok"),
            method=result.get("method"),
            mapped=len(result.get("mappings") or [])
            - len(result.get("unmapped_indices") or []),
            unmapped=len(result.get("unmapped_indices") or []),
            label_pool_source=result.get("label_pool_source"),
            hint=result.get("hint"),
        )
        return {"data": result}
    except HTTPException:
        raise
    except Exception as exc:
        raise _http_from_llm_error(exc) from exc


@router.post("/judge-detection-labels", summary="整图评分复核检测框标签")
async def api_judge_detection_labels(
    body: JudgeDetectionLabelsRequest,
    current_user: CurrentUser,
    db: DbSession,
    settings: SettingsDep,
):
    try:
        if not settings.annotation_judge_enabled:
            return {
                "data": {
                    "ok": True,
                    "verdict": "accept",
                    "confidence": 1.0,
                    "summary": "JudgeAgent 未启用，直接通过。",
                    "issues": [],
                    "retry_feedback": "",
                    "checked_boxes": len(body.boxes),
                }
            }
        is_direct = bool(body.api_key.strip() and body.base_url.strip() and body.model.strip())
        if is_direct:
            provider_is_vision = body.supports_vision
        else:
            svc = LlmProviderService(db, settings)
            row = await svc.get_for_user(uuid.UUID(body.provider_id), current_user.id)
            if row is None:
                raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="llm_provider_not_found")
            provider_is_vision = await svc.ensure_vision_probed(row)
        if not provider_is_vision:
            return {
                "data": {
                    "ok": False,
                    "error": "vision_not_supported",
                    "verdict": "reject",
                    "confidence": 0.0,
                    "summary": "评分子 Agent 需要多模态视觉模型。",
                    "issues": [],
                    "retry_feedback": "请使用通过视觉探针的多模态模型。",
                    "checked_boxes": 0,
                }
            }
        llm = await _llm_for_provider(
            db,
            settings,
            current_user.id,
            body.provider_id,
            temperature=settings.annotation_judge_temperature,
            api_key_direct=body.api_key,
            base_url_direct=body.base_url,
            model_direct=body.model,
        )
        result = await judge_detection_labels(
            llm,
            user_request=body.user_request,
            intent_summary=body.intent_summary,
            label_candidates=body.label_candidates,
            boxes=body.boxes,
            mappings=body.mappings,
            annotations=body.annotations,
            image_absolute_path=body.image_absolute_path,
            image_base64=body.image_base64,
            attempt=body.attempt,
            max_retries=body.max_retries,
            settings=settings,
        )
        return {"data": result}
    except HTTPException:
        raise
    except Exception as exc:
        raise _http_from_llm_error(exc) from exc
