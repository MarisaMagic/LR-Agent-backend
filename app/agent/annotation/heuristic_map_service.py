"""启发式检测框 → 标签映射（无 LLM）。

作为 map_detection_boxes_to_labels_unified 的非视觉回退路径，
也可通过 /map-heuristic API 独立调用。匹配规则：
  1. 检测类名与标签名精确匹配
  2. 检测类名与标签名部分匹配（唯一命中时）
  3. OCR 文本含唯一标签名
"""

from __future__ import annotations


def heuristic_map_boxes(
    boxes: list[dict],
    label_candidates: list[dict],
    *,
    ocr_text: str = "",
) -> list[dict]:
    """为每个检测框尝试基于类名/OCR 的自动标签映射，无法匹配时 label_id 为空。"""
    by_name: dict[str, dict] = {}
    for item in label_candidates:
        name = str(item.get("name") or "").strip().lower()
        if name:
            by_name[name] = item

    flat = list(label_candidates)
    ocr_lower = (ocr_text or "").lower()
    mappings: list[dict] = []

    for idx, box in enumerate(boxes):
        det = str(box.get("class_name") or box.get("detection_label") or "").strip()
        chosen: dict | None = None
        reason = ""

        if det:
            hit = by_name.get(det.lower())
            if hit:
                chosen = hit
                reason = f"检测类名 {det!r} 与标签匹配"
            else:
                det_lower = det.lower()
                partial = [
                    n
                    for n in flat
                    if det_lower in str(n.get("name") or "").lower()
                    or str(n.get("name") or "").lower() in det_lower
                ]
                if len(partial) == 1:
                    chosen = partial[0]
                    reason = f"检测类名与标签 {chosen.get('name')!r} 部分匹配"

        if not chosen and ocr_lower:
            name_hits = [
                n for n in flat if str(n.get("name") or "").lower() in ocr_lower
            ]
            if len(name_hits) == 1:
                chosen = name_hits[0]
                reason = f"OCR 含标签名 {chosen.get('name')!r}"

        mappings.append(
            {
                "box_index": int(box.get("box_index", idx)),
                "label_id": str(chosen.get("id") or "") if chosen else "",
                "reason": reason or ("未能自动映射" if not chosen else reason),
            }
        )
    return mappings
