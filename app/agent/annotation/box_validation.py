"""BBox finalize validation (fusion box_labels parity, flat label list)."""
from __future__ import annotations


def validate_mappings_for_finalize(
    boxes: list[dict],
    mappings: list[dict],
    *,
    valid_label_ids: set[str],
    tolerance: float = 0.02,
) -> dict:
    errors: list[str] = []
    if not mappings:
        errors.append("mappings 为空")
        return {"valid": False, "errors": errors}

    mapped_by_index = {
        int(m.get("box_index", -1)): m for m in mappings if isinstance(m, dict)
    }
    labeled = 0
    for box in boxes:
        idx = int(box.get("box_index", -1))
        m = mapped_by_index.get(idx)
        if not m:
            continue
        lid = str(m.get("label_id") or "").strip()
        if not lid:
            continue
        if lid not in valid_label_ids:
            errors.append(f"box_index={idx} 的 label_id 不在候选中")
            continue
        labeled += 1

    if labeled == 0:
        errors.append("无有效 label_id 映射")
    return {"valid": not errors, "errors": errors, "labeled_count": labeled}
