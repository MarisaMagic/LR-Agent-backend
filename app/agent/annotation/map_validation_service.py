"""视觉映射结果校验与重试辅助（规则驱动，标签名来自候选集）。"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum


class MappingIssueCode(str, Enum):
    DUPLICATE_LABEL = "duplicate_label"
    REASON_NAME_MISMATCH = "reason_name_mismatch"
    REASON_CONTRADICTS_LABEL = "reason_contradicts_label"
    INVALID_LABEL_ID = "invalid_label_id"


@dataclass
class MappingIssue:
    box_index: int
    code: MappingIssueCode
    message: str


@dataclass
class ValidationResult:
    issues: list[MappingIssue] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.issues

    def issues_for_box(self, box_index: int) -> list[MappingIssue]:
        return [i for i in self.issues if i.box_index == box_index]


_REASON_CONTRADICT_RE = re.compile(
    r"无法确认|无法确定|无法选择|不能确定|不符|不匹配|无关|"
    r"cannot determine|not sure|no match|does not match|unable to",
    re.IGNORECASE,
)


def _name_to_id(candidates: list[dict]) -> dict[str, str]:
    out: dict[str, str] = {}
    for c in candidates:
        name = str(c.get("name") or "").strip().lower()
        cid = str(c.get("id") or "").strip()
        if name and cid:
            out[name] = cid
    return out


def find_label_names_in_text(text: str, candidates: list[dict]) -> list[str]:
    """在 reason 等文本中查找出现的候选标签名（大小写不敏感子串）。"""
    raw = (text or "").lower()
    if not raw:
        return []
    hits: list[str] = []
    seen: set[str] = set()
    for c in candidates:
        name = str(c.get("name") or "").strip()
        if not name:
            continue
        key = name.lower()
        if key in seen:
            continue
        if key in raw:
            seen.add(key)
            hits.append(name)
    return hits


def format_issues_for_retry(issues: list[MappingIssue]) -> str:
    if not issues:
        return ""
    return "；".join(f"[{i.code}] {i.message}" for i in issues)


def candidates_for_retry_box(
    candidates: list[dict],
    used_label_ids: set[str],
    *,
    keep_label_id: str = "",
) -> list[dict]:
    """重试时排除同图已被其它框占用的 label_id（L3 占用约束）。"""
    if not used_label_ids:
        return candidates
    keep = keep_label_id.strip()
    filtered = [
        c
        for c in candidates
        if str(c.get("id") or "") not in used_label_ids
        or str(c.get("id") or "") == keep
    ]
    return filtered if filtered else candidates


def validate_vision_mappings(
    mappings: list[dict],
    candidates: list[dict],
    *,
    instance_labels: bool,
    valid_ids: set[str] | None = None,
) -> ValidationResult:
    """校验逐框视觉映射结果。"""
    ids = valid_ids or {str(c.get("id") or "") for c in candidates}
    name_to_id = _name_to_id(candidates)
    issues: list[MappingIssue] = []

    by_box: dict[int, dict] = {}
    for m in mappings:
        by_box[int(m.get("box_index", 0))] = m

    label_to_boxes: dict[str, list[int]] = {}
    for box_index, m in by_box.items():
        lid = str(m.get("label_id") or "").strip()
        if not lid:
            continue
        if lid not in ids:
            issues.append(
                MappingIssue(
                    box_index=box_index,
                    code=MappingIssueCode.INVALID_LABEL_ID,
                    message=f"label_id={lid!r} 不在候选中",
                )
            )
            continue
        label_to_boxes.setdefault(lid, []).append(box_index)

    if instance_labels:
        for lid, box_indices in label_to_boxes.items():
            if len(box_indices) <= 1:
                continue
            name = next(
                (str(c.get("name") or lid) for c in candidates if str(c.get("id") or "") == lid),
                lid,
            )
            for box_index in box_indices:
                issues.append(
                    MappingIssue(
                        box_index=box_index,
                        code=MappingIssueCode.DUPLICATE_LABEL,
                        message=f"标签 {name!r} 被分配给多个框 {box_indices}",
                    )
                )

    for box_index, m in by_box.items():
        lid = str(m.get("label_id") or "").strip()
        reason = str(m.get("reason") or "")

        if lid and _REASON_CONTRADICT_RE.search(reason):
            issues.append(
                MappingIssue(
                    box_index=box_index,
                    code=MappingIssueCode.REASON_CONTRADICTS_LABEL,
                    message="reason 表示无法/不匹配，但返回了非空 label_id",
                )
            )

        if not lid or not reason:
            continue

        mentioned = find_label_names_in_text(reason, candidates)
        if len(mentioned) != 1:
            continue
        mentioned_name = mentioned[0].lower()
        expected_id = name_to_id.get(mentioned_name, "")
        if expected_id and expected_id != lid:
            chosen_name = next(
                (str(c.get("name") or "") for c in candidates if str(c.get("id") or "") == lid),
                lid,
            )
            issues.append(
                MappingIssue(
                    box_index=box_index,
                    code=MappingIssueCode.REASON_NAME_MISMATCH,
                    message=(
                        f"reason 指向 {mentioned[0]!r}，"
                        f"但 label_id 对应 {chosen_name!r}"
                    ),
                )
            )

    return ValidationResult(issues=issues)
