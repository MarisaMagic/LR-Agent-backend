"""Internal structured outputs for annotation agent LLM calls."""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class AnnotationScopePayload(BaseModel):
    scope_summary: str = ""
    include_detection_labels: list[str] = Field(default_factory=list)
    exclude_detection_labels: list[str] = Field(default_factory=list)
    include_label_names: list[str] = Field(default_factory=list)
    exclude_label_names: list[str] = Field(default_factory=list)


class DetectionHintsPayload(BaseModel):
    needs_object_detection: bool = True
    model_id: str | None = None
    conf_threshold: float | None = None
    iou_threshold: float | None = None
    notes: str = ""


class SubAgentConstraintsPayload(BaseModel):
    require_per_box_mapping: bool = True
    allow_unlabeled_boxes: bool = False
    min_labeled_box_count: int = 1


class BatchPlanResult(BaseModel):
    intent_summary: str = "批量图片标注"
    label_strategy: Literal["map_each_box_to_label", "single_label_for_all_boxes"] = (
        "map_each_box_to_label"
    )
    use_vision_mapping: bool = False
    detection_hints: DetectionHintsPayload = Field(default_factory=DetectionHintsPayload)
    sub_agent_constraints: SubAgentConstraintsPayload = Field(
        default_factory=SubAgentConstraintsPayload
    )
    annotation_scope: AnnotationScopePayload = Field(default_factory=AnnotationScopePayload)
    plan_steps: list[str] = Field(default_factory=list)


class BatchPrepareResult(BaseModel):
    """Scope + plan in one LLM call."""

    selected_paths: list[str] = Field(default_factory=list)
    scope_reason: str = ""
    plan: BatchPlanResult = Field(default_factory=BatchPlanResult)

