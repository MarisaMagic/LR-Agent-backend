"""Annotation scope filtering & merge (fusion-reasoning parity)."""
from __future__ import annotations

import re
from typing import Any

from pydantic import BaseModel, Field, field_validator

from app.agent.annotation.schemas import AnnotationScopePayload

SCOPE_KEYWORD_TO_DETECTION: dict[str, list[str]] = {
    "人脸": ["person", "face"],
    "面部": ["person", "face"],
    "人体": ["person"],
    "人物": ["person"],
    "人": ["person"],
    "球员": ["person"],
    "篮球": ["sports ball", "basketball"],
    "球": ["sports ball", "basketball", "ball"],
    "汽车": ["car"],
    "车": ["car"],
    "猫": ["cat"],
    "狗": ["dog"],
}

ONLY_ANNOTATE_RE = re.compile(
    r"(?:只|仅)(?:标注|标|检测|框选|识别|打框|标出)?(.{1,24})"
)
EXCLUDE_RE = re.compile(
    r"(?:不要|别|排除|不包括|无需|不用)(?:标注|标|检测|框选)?(.{1,24})"
)


class AnnotationScope(BaseModel):
    scope_summary: str = ""
    include_detection_labels: list[str] = Field(default_factory=list)
    exclude_detection_labels: list[str] = Field(default_factory=list)
    include_label_names: list[str] = Field(default_factory=list)
    exclude_label_names: list[str] = Field(default_factory=list)

    @field_validator(
        "include_detection_labels",
        "exclude_detection_labels",
        "include_label_names",
        "exclude_label_names",
        mode="before",
    )
    @classmethod
    def _coerce_str_list(cls, v: Any) -> list[str]:
        if v is None:
            return []
        if isinstance(v, str):
            v = v.strip()
            return [v] if v else []
        if isinstance(v, list):
            return [str(item).strip() for item in v if str(item).strip()]
        return []

    def is_restricted(self) -> bool:
        return bool(
            self.scope_summary.strip()
            or self.include_detection_labels
            or self.exclude_detection_labels
            or self.include_label_names
            or self.exclude_label_names
        )

    def to_payload(self) -> AnnotationScopePayload:
        return AnnotationScopePayload(
            scope_summary=self.scope_summary,
            include_detection_labels=list(self.include_detection_labels),
            exclude_detection_labels=list(self.exclude_detection_labels),
            include_label_names=list(self.include_label_names),
            exclude_label_names=list(self.exclude_label_names),
        )

    @classmethod
    def from_payload(cls, payload: AnnotationScopePayload | dict[str, Any] | None) -> AnnotationScope:
        if payload is None:
            return cls()
        if isinstance(payload, AnnotationScopePayload):
            return cls.model_validate(payload.model_dump())
        if isinstance(payload, dict):
            return cls.model_validate(payload)
        return cls()


def normalize_detection_label(label: str) -> str:
    return re.sub(r"\s+", " ", (label or "").strip().lower().replace("_", " "))


def expand_detection_aliases(terms: list[str]) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for term in terms:
        t = (term or "").strip()
        if not t:
            continue
        for c in SCOPE_KEYWORD_TO_DETECTION.get(t, [t]):
            nc = normalize_detection_label(c)
            if nc and nc not in seen:
                seen.add(nc)
                out.append(c)
    return out


def _extract_scope_terms(text: str, pattern: re.Pattern[str]) -> list[str]:
    terms: list[str] = []
    for m in pattern.finditer(text):
        chunk = (m.group(1) or "").strip()
        chunk = re.split(r"[，。；;、和与及\s]+", chunk)[0].strip()
        if chunk and len(chunk) <= 24:
            terms.append(chunk)
    return terms


def infer_annotation_scope_from_text(
    user_text: str,
    *,
    label_names: list[str] | None = None,
) -> AnnotationScope:
    text = (user_text or "").strip()
    if not text:
        return AnnotationScope()

    include_terms = _extract_scope_terms(text, ONLY_ANNOTATE_RE)
    exclude_terms = _extract_scope_terms(text, EXCLUDE_RE)
    scope = AnnotationScope()
    include_det: list[str] = []
    exclude_det: list[str] = []
    include_labels: list[str] = []

    for term in include_terms:
        include_det.extend(expand_detection_aliases([term]))
        if term in SCOPE_KEYWORD_TO_DETECTION or term in ("人脸", "面部", "人物", "人体"):
            scope.scope_summary = f"仅标注{term}"

    if not include_terms:
        if any(k in text for k in ("人脸", "面部")):
            include_det.extend(expand_detection_aliases(["人脸"]))
            scope.scope_summary = scope.scope_summary or "仅标注人脸/人物"
        elif "人物" in text and "标注" in text:
            include_det.extend(expand_detection_aliases(["人物"]))
            scope.scope_summary = scope.scope_summary or "仅标注人物"

    for term in exclude_terms:
        exclude_det.extend(expand_detection_aliases([term]))
        if not scope.scope_summary:
            scope.scope_summary = f"排除{term}"

    if any(k in text for k in ("篮球", "球")) and any(
        k in text for k in ("不要", "别", "排除", "不包括", "无需")
    ):
        exclude_det.extend(expand_detection_aliases(["篮球"]))

    if label_names:
        for name in label_names:
            if name and name in text and any(k in text for k in ("只", "仅", "标注")):
                include_labels.append(name)

    scope.include_detection_labels = list(dict.fromkeys(include_det))
    scope.exclude_detection_labels = list(dict.fromkeys(exclude_det))
    scope.include_label_names = list(dict.fromkeys(include_labels))
    return scope


def merge_annotation_scope(
    scope: AnnotationScopePayload | AnnotationScope | None,
    user_text: str,
    *,
    label_names: list[str] | None = None,
) -> AnnotationScope:
    base = AnnotationScope.from_payload(scope) if scope else AnnotationScope()
    inferred = infer_annotation_scope_from_text(user_text, label_names=label_names)
    if not base.scope_summary.strip() and inferred.scope_summary.strip():
        base.scope_summary = inferred.scope_summary
    if not base.include_detection_labels and inferred.include_detection_labels:
        base.include_detection_labels = inferred.include_detection_labels
    if not base.exclude_detection_labels and inferred.exclude_detection_labels:
        base.exclude_detection_labels = inferred.exclude_detection_labels
    if not base.include_label_names and inferred.include_label_names:
        base.include_label_names = inferred.include_label_names
    if not base.exclude_label_names and inferred.exclude_label_names:
        base.exclude_label_names = inferred.exclude_label_names
    return base


def filter_label_candidates_by_scope(
    candidates: list[dict],
    scope: AnnotationScope,
) -> list[dict]:
    if not scope.include_label_names and not scope.exclude_label_names:
        return candidates
    include = {n.lower() for n in scope.include_label_names}
    exclude = {n.lower() for n in scope.exclude_label_names}
    filtered: list[dict] = []
    for c in candidates:
        name = str(c.get("name") or "").lower()
        if exclude and name in exclude:
            continue
        if include:
            if name in include:
                filtered.append(c)
        else:
            filtered.append(c)
    return filtered if filtered else candidates
