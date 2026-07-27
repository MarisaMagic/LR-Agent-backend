import logging

from fastapi import APIRouter, HTTPException, status
from langchain_openai import ChatOpenAI
from openai import BadRequestError

from app.agent.annotation import (
    heuristic_map_boxes,
    judge_detection_labels,
    map_detection_boxes_to_labels_unified,
    prepare_batch_annotation,
)
from app.agent.annotation.debug_log import log_annotation_agent
from app.agent.annotation.image_bytes_loader import load_image_bytes
from app.agent.annotation.mutation_prepare_service import prepare_mutation_annotation
from app.agent.annotation.schemas import AnnotationScopePayload
from app.core.deps import CurrentUser, SettingsDep
from app.schemas.annotation_agent import (
    BatchPrepareRequest,
    HeuristicMapRequest,
    JudgeDetectionLabelsRequest,
    MapDetectionBoxesRequest,
    MutationPrepareRequest,
)

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


def _require_direct_llm(
    *,
    api_key: str,
    base_url: str,
    model: str,
    temperature: float,
    streaming: bool = False,
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
        streaming=streaming,
        temperature=temperature,
        timeout=120,
    )


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
    settings: SettingsDep,
):
    del current_user
    if not settings.agent_mutation_enabled:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="agent_mutation_disabled",
        )
    try:
        llm = _require_direct_llm(
            api_key=body.api_key,
            base_url=body.base_url,
            model=body.model,
            temperature=settings.annotation_prepare_temperature,
        )
        label_names = None
        if body.project is not None:
            label_names = [
                str(label.get("name") or "")
                for label in (body.project.labels or [])
                if str(label.get("name") or "").strip()
            ]

        result = await prepare_mutation_annotation(
            llm,
            user_request=body.user_request,
            current_relative_path=(body.current_relative_path or "").strip(),
            candidates=[candidate.model_dump() for candidate in body.candidates],
            label_names=label_names or [],
            selected_annotation_ids=body.selected_annotation_ids or None,
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
    settings: SettingsDep,
):
    del current_user
    try:
        llm = _require_direct_llm(
            api_key=body.api_key,
            base_url=body.base_url,
            model=body.model,
            temperature=settings.annotation_prepare_temperature,
        )
        provider_is_vision = body.supports_vision

        label_names = None
        project_name = None
        if body.project is not None:
            project_name = body.project.name or None
            label_names = [
                str(label.get("name") or "")
                for label in (body.project.labels or [])
                if str(label.get("name") or "").strip()
            ]

        current_rel = (body.current_relative_path or "").strip()
        candidates = [candidate.model_dump() for candidate in body.candidates]
        result = await prepare_batch_annotation(
            llm,
            user_request=body.user_request,
            current_relative_path=current_rel,
            candidates=candidates,
            label_candidates=body.label_candidates,
            detection_models=body.detection_models,
            default_conf=body.default_conf_threshold,
            default_iou=body.default_iou_threshold,
            provider_is_vision=provider_is_vision,
            project_name=project_name,
            label_names=label_names,
            preselected_paths=body.preselected_paths or None,
        )
        payload = {
            "selected_paths": result.selected_paths,
            "scope_reason": result.scope_reason,
            "resolved_user_request": body.user_request,
            "judge_config": {
                "enabled": settings.annotation_judge_enabled and provider_is_vision,
                "max_retries": settings.annotation_judge_max_retries,
                "reject_submit_partial": settings.annotation_judge_reject_submit_partial,
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
    settings: SettingsDep,
):
    del current_user
    try:
        provider_is_vision = body.supports_vision
        use_vision_requested = bool(body.use_vision)
        use_vision = use_vision_requested and provider_is_vision
        llm = None
        if use_vision:
            llm = _require_direct_llm(
                api_key=body.api_key,
                base_url=body.base_url,
                model=body.model,
                temperature=settings.annotation_llm_temperature,
            )
        log_annotation_agent(
            "map-api",
            "map-detection-boxes 请求",
            provider_id=body.provider_id,
            provider_name=body.model,
            provider_model=body.model,
            provider_is_vision=provider_is_vision,
            vision_probe_detail="direct",
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
            label_names=[
                str(candidate.get("name") or "")
                for candidate in (body.label_candidates or [])[:20]
            ],
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
    settings: SettingsDep,
):
    del current_user
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
        provider_is_vision = body.supports_vision
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
        llm = _require_direct_llm(
            api_key=body.api_key,
            base_url=body.base_url,
            model=body.model,
            temperature=settings.annotation_judge_temperature,
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
