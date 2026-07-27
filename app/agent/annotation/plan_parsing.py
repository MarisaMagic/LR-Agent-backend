"""Shared parsing of LLM JSON into BatchPlanResult."""
from __future__ import annotations

from typing import Any

from app.agent.annotation.debug_log import log_annotation_agent
from app.agent.annotation.label_vision_policy import labels_require_vision_mapping
from app.agent.annotation.schemas import (
    AnnotationScopePayload,
    BatchPlanResult,
    DetectionHintsPayload,
    SubAgentConstraintsPayload,
)


def build_batch_plan_from_data(
    data: dict[str, Any],
    *,
    user_request: str,
    intent_summary: str,
    annotation_scope: AnnotationScopePayload,
    label_candidates: list[dict],
    detection_models: list[dict],
    default_conf: float,
    default_iou: float,
    provider_is_vision: bool,
    log_prefix: str = "batch-prepare",
) -> BatchPlanResult:
    label_strategy = str(data.get("label_strategy") or "map_each_box_to_label")
    if label_strategy not in ("single_label_for_all_boxes", "map_each_box_to_label"):
        label_strategy = "map_each_box_to_label"

    det_raw = data.get("detection_hints") if isinstance(data.get("detection_hints"), dict) else {}
    valid_ids = {str(m.get("id") or "") for m in detection_models}
    model_id = str(det_raw.get("model_id") or "").strip()
    if model_id and model_id not in valid_ids:
        model_id = ""

    def _clamp(val: Any, lo: float, hi: float, default: float) -> float:
        try:
            f = float(val)
            return max(lo, min(hi, f))
        except (TypeError, ValueError):
            return default

    conf = det_raw.get("conf_threshold")
    iou = det_raw.get("iou_threshold")
    detection_hints = DetectionHintsPayload(
        needs_object_detection=bool(det_raw.get("needs_object_detection", True)),
        model_id=model_id or None,
        conf_threshold=_clamp(conf, 0.05, 0.95, default_conf) if conf is not None else default_conf,
        iou_threshold=_clamp(iou, 0.05, 0.95, default_iou) if iou is not None else default_iou,
        notes=str(det_raw.get("notes") or "")[:200],
    )

    sub_raw = data.get("sub_agent_constraints") if isinstance(data.get("sub_agent_constraints"), dict) else {}
    instance_labels = labels_require_vision_mapping(label_candidates)
    default_allow_unlabeled = instance_labels
    sub_constraints = SubAgentConstraintsPayload(
        require_per_box_mapping=bool(
            sub_raw.get("require_per_box_mapping", label_strategy == "map_each_box_to_label")
        ),
        allow_unlabeled_boxes=bool(
            sub_raw.get("allow_unlabeled_boxes", default_allow_unlabeled)
        ),
        min_labeled_box_count=max(1, int(sub_raw.get("min_labeled_box_count", 1))),
    )

    scope_raw = data.get("annotation_scope") if isinstance(data.get("annotation_scope"), dict) else {}
    merged_scope = AnnotationScopePayload(
        scope_summary=str(scope_raw.get("scope_summary") or annotation_scope.scope_summary),
        include_detection_labels=list(
            scope_raw.get("include_detection_labels") or annotation_scope.include_detection_labels
        ),
        exclude_detection_labels=list(
            scope_raw.get("exclude_detection_labels") or annotation_scope.exclude_detection_labels
        ),
        include_label_names=list(
            scope_raw.get("include_label_names") or annotation_scope.include_label_names
        ),
        exclude_label_names=list(
            scope_raw.get("exclude_label_names") or annotation_scope.exclude_label_names
        ),
    )

    plan_steps = data.get("plan_steps") if isinstance(data.get("plan_steps"), list) else []
    steps = [str(s).strip() for s in plan_steps if str(s).strip()][:12]

    plan_use_vision_raw = bool(data.get("use_vision_mapping", False))
    task_needs_vision = (
        label_strategy == "map_each_box_to_label"
        and labels_require_vision_mapping(label_candidates)
    )
    want_vision = plan_use_vision_raw or task_needs_vision
    use_vision = want_vision and bool(provider_is_vision)
    if want_vision and not provider_is_vision:
        log_annotation_agent(
            f"{log_prefix}-gate",
            "需要视觉映射但提供商视觉探针未通过",
            provider_is_vision=provider_is_vision,
            task_needs_vision=task_needs_vision,
        )
    elif task_needs_vision and provider_is_vision and not plan_use_vision_raw:
        log_annotation_agent(
            f"{log_prefix}-auto",
            "标签与 YOLO 类名不匹配，已自动开启 use_vision_mapping",
            label_names=[str(c.get("name") or "") for c in label_candidates[:20]],
        )

    return BatchPlanResult(
        intent_summary=str(data.get("intent_summary") or intent_summary or user_request[:200] or "批量图片标注"),
        label_strategy=label_strategy,  # type: ignore[arg-type]
        use_vision_mapping=use_vision,
        detection_hints=detection_hints,
        sub_agent_constraints=sub_constraints,
        annotation_scope=merged_scope,
        plan_steps=steps,
    )
