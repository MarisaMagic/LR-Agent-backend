from typing import Any

from pydantic import BaseModel, Field


class AnnotationProjectSnapshotInput(BaseModel):
    project_id: str = Field(min_length=1, max_length=64)
    name: str = ""
    modality: str = "image"
    annotation_type: str = "bbox"
    labels: list[dict[str, Any]] = Field(default_factory=list)


class ImageCandidateInput(BaseModel):
    relative_path: str
    name: str = ""
    parent: str = ""
    index: int = 0


class AnnotationLlmBaseRequest(BaseModel):
    provider_id: str = Field(min_length=1, max_length=64)


class BatchPrepareRequest(AnnotationLlmBaseRequest):
    """Single-shot scope + plan (replaces separate parse-task / parse-scope / create-plan in batch)."""

    user_request: str = Field(min_length=1, max_length=20_000)
    preselected_paths: list[str] = Field(
        default_factory=list,
        description="客户端显式指定的图片路径（如 UI 勾选）；非空则跳过 LLM 选图，仅生成执行计划",
    )
    session_id: str | None = Field(default=None, max_length=64)
    current_relative_path: str = ""
    candidates: list[ImageCandidateInput] = Field(default_factory=list)
    label_candidates: list[dict[str, Any]] = Field(default_factory=list)
    detection_models: list[dict[str, Any]] = Field(default_factory=list)
    default_conf_threshold: float = Field(default=0.7, ge=0.05, le=0.95)
    default_iou_threshold: float = Field(default=0.5, ge=0.05, le=0.95)
    project: AnnotationProjectSnapshotInput | None = None


class HeuristicMapRequest(AnnotationLlmBaseRequest):
    boxes: list[dict[str, Any]] = Field(default_factory=list)
    label_candidates: list[dict[str, Any]] = Field(default_factory=list)
    ocr_text: str = ""


class MapDetectionBoxesRequest(AnnotationLlmBaseRequest):
    """Fusion-style unified mapping (heuristic or vision_crop only)."""

    user_request: str = Field(default="", max_length=20_000)
    intent_summary: str = ""
    label_candidates: list[dict[str, Any]] = Field(default_factory=list)
    boxes: list[dict[str, Any]] = Field(default_factory=list)
    use_vision: bool = False
    label_strategy: str = "map_each_box_to_label"
    single_label_id: str | None = None
    annotation_scope: dict[str, Any] = Field(default_factory=dict)
    ocr_text: str = ""
    image_absolute_path: str = Field(default="", max_length=1024)
    image_base64: str = Field(default="", max_length=16_000_000)
    mime_type: str = "image/jpeg"


