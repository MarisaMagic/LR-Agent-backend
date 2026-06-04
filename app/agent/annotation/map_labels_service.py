"""Unified detection-box → label mapping (fusion image_box_labels parity)."""
from __future__ import annotations

import asyncio
import base64
import io
import json
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

from app.agent.annotation.annotation_scope import (
    AnnotationScope,
    filter_label_candidates_by_scope,
)
from app.agent.annotation.heuristic_map_service import heuristic_map_boxes
from app.agent.annotation.image_bytes_loader import load_image_bytes
from app.agent.annotation.json_utils import extract_json_object
from app.agent.annotation.debug_log import log_annotation_agent
from app.agent.annotation.schemas import AnnotationScopePayload
from app.core.config import get_settings

CROP_VISION_SYSTEM = """你是视觉标注助手。你会收到一张裁剪后的目标区域图。
请根据图像内容与用户标注意图，从标签候选中选择最合适的 label_id。
若裁剪区域与任一候选标签语义不符，必须返回空 label_id。
只输出 JSON：{"label_id":"...","reason":"..."} ；无法确定或与任务无关时 label_id 必须为 ""。"""


def _resize_image_for_llm(img, max_side: int = 768):
    from PIL import Image

    w, h = img.size
    if max(w, h) <= max_side:
        return img
    scale = max_side / max(w, h)
    return img.resize((int(w * scale), int(h * scale)), Image.Resampling.LANCZOS)


def _pil_to_data_url(img, fmt: str = "JPEG") -> str:
    buf = io.BytesIO()
    if fmt.upper() == "JPEG" and img.mode in ("RGBA", "P"):
        img = img.convert("RGB")
    img.save(buf, format=fmt, quality=85)
    mime = "image/jpeg" if fmt.upper() == "JPEG" else f"image/png"
    b64 = base64.b64encode(buf.getvalue()).decode("ascii")
    return f"data:{mime};base64,{b64}"


def _coords_are_normalized(boxes: list[dict]) -> bool:
    for b in boxes:
        for key in ("x", "y", "width", "height"):
            try:
                if float(b.get(key, 0)) > 1.5:
                    return False
            except (TypeError, ValueError):
                return False
    return bool(boxes)


def _crop_boxes_from_image_bytes(
    image_bytes: bytes,
    boxes: list[dict],
    *,
    normalized: bool,
) -> list[dict[str, Any]]:
    from PIL import Image

    regions: list[dict[str, Any]] = []
    try:
        with Image.open(io.BytesIO(image_bytes)) as original:
            img = original.convert("RGB")
            iw, ih = img.size
            for idx, box in enumerate(boxes):
                try:
                    x = float(box.get("x", 0))
                    y = float(box.get("y", 0))
                    w = float(box.get("width", 0))
                    h = float(box.get("height", 0))
                except (TypeError, ValueError):
                    continue
                if w <= 0 or h <= 0:
                    continue
                if normalized:
                    x, y, w, h = x * iw, y * ih, w * iw, h * ih
                left = max(0, min(int(x), iw - 1))
                top = max(0, min(int(y), ih - 1))
                right = max(left + 1, min(int(x + w), iw))
                bottom = max(top + 1, min(int(y + h), ih))
                crop = img.crop((left, top, right, bottom))
                regions.append(
                    {
                        "box_index": int(box.get("box_index", idx)),
                        "data_url": _pil_to_data_url(_resize_image_for_llm(crop, 768)),
                        "box": box,
                    }
                )
    except Exception:
        return []
    return regions


async def _vision_map_box_with_crop(
    llm: ChatOpenAI,
    *,
    box_index: int,
    box: dict,
    crop_data_url: str,
    candidates: list[dict],
    user_request: str,
    intent_summary: str,
    scope_note: str = "",
) -> dict[str, Any]:
    valid_ids = {str(c.get("id") or "") for c in candidates}
    user_text = (
        f"用户请求：{user_request}\n意图：{intent_summary}\n"
        f"box_index={box_index} 检测类名：{box.get('class_name') or box.get('detection_label') or ''}\n"
        f"{scope_note}\n\nlabel_candidates:\n"
        f"{json.dumps(candidates[:40], ensure_ascii=False)}"
    )
    human_content: Any = [
        {"type": "text", "text": user_text},
        {"type": "image_url", "image_url": {"url": crop_data_url}},
    ]
    messages = [
        SystemMessage(content=CROP_VISION_SYSTEM),
        HumanMessage(content=human_content),
    ]
    try:
        resp = await llm.ainvoke(messages)
    except Exception:
        return {"box_index": box_index, "label_id": "", "reason": "视觉调用失败"}
    content = resp.content if hasattr(resp, "content") else str(resp)
    if isinstance(content, list):
        content = "".join(
            part.get("text", "") if isinstance(part, dict) else str(part)
            for part in content
        )
    parsed = extract_json_object(str(content))
    lid = str(parsed.get("label_id") or "").strip()
    if lid and lid not in valid_ids:
        lid = ""
    return {
        "box_index": box_index,
        "label_id": lid,
        "reason": str(parsed.get("reason") or ""),
    }


async def _vision_map_regions_concurrent(
    llm: ChatOpenAI,
    regions: list[dict[str, Any]],
    *,
    candidates: list[dict],
    user_request: str,
    intent_summary: str,
    scope_note: str,
    concurrency: int,
) -> list[dict[str, Any]]:
    sem = asyncio.Semaphore(max(1, concurrency))

    async def _map_one(region: dict[str, Any]) -> dict[str, Any]:
        async with sem:
            return await _vision_map_box_with_crop(
                llm,
                box_index=int(region["box_index"]),
                box=region["box"],
                crop_data_url=str(region["data_url"]),
                candidates=candidates,
                user_request=user_request,
                intent_summary=intent_summary,
                scope_note=scope_note,
            )

    return list(await asyncio.gather(*[_map_one(r) for r in regions]))


async def map_detection_boxes_to_labels_unified(
    llm: ChatOpenAI | None,
    *,
    user_request: str,
    intent_summary: str,
    label_candidates: list[dict],
    boxes: list[dict],
    use_vision: bool = False,
    ocr_text: str = "",
    scope: AnnotationScopePayload | AnnotationScope | None = None,
    label_strategy: str = "map_each_box_to_label",
    single_label_id: str | None = None,
    image_absolute_path: str = "",
    image_base64: str = "",
    mime_type: str = "image/jpeg",
    vision_map_concurrency: int | None = None,
) -> dict[str, Any]:
    """
    Fusion-style mapping: vision_crop when use_vision and llm+image present;
    otherwise heuristic only (no text-only LLM fallback).
    """
    scope_model = (
        scope
        if isinstance(scope, AnnotationScope)
        else AnnotationScope.from_payload(scope)
    )
    candidates = filter_label_candidates_by_scope(list(label_candidates), scope_model)
    valid_ids = {str(c.get("id") or "") for c in candidates}
    image_bytes, image_source = load_image_bytes(
        image_absolute_path=image_absolute_path,
        image_base64=image_base64,
    )
    concurrency = vision_map_concurrency
    if concurrency is None:
        concurrency = get_settings().annotation_vision_map_concurrency

    log_annotation_agent(
        "map-start",
        "map_detection_boxes_to_labels_unified",
        use_vision=use_vision,
        box_count=len(boxes),
        label_candidate_count=len(label_candidates),
        label_strategy=label_strategy,
        has_image_bytes=image_bytes is not None,
        image_source=image_source,
        vision_map_concurrency=concurrency if use_vision else None,
        scope_restricted=scope_model.is_restricted(),
        scope_summary=scope_model.scope_summary,
    )

    if not boxes:
        return {"ok": False, "error": "boxes 不能为空", "method": "none"}

    if label_strategy == "single_label_for_all_boxes" and single_label_id:
        if single_label_id in valid_ids:
            mappings = [
                {
                    "box_index": int(b.get("box_index", i)),
                    "label_id": single_label_id,
                    "reason": "single_label_for_all_boxes",
                }
                for i, b in enumerate(boxes)
            ]
            return {
                "ok": True,
                "method": "single_label",
                "mappings": mappings,
                "unmapped_indices": [],
            }

    normalized_boxes = []
    for i, b in enumerate(boxes):
        normalized_boxes.append(
            {
                "box_index": int(b.get("box_index", i)),
                "x": float(b.get("x", 0)),
                "y": float(b.get("y", 0)),
                "width": float(b.get("width", 0)),
                "height": float(b.get("height", 0)),
                "class_name": b.get("class_name") or b.get("detection_label"),
                "detection_label": b.get("detection_label") or b.get("class_name"),
                "confidence": b.get("confidence"),
            }
        )

    if use_vision and llm is not None:
        if image_bytes is None:
            return {
                "ok": False,
                "error": "image_unavailable",
                "method": "vision_crop",
                "hint": "视觉映射需要可读本地 image_absolute_path 或 image_base64",
            }

        scope_note = ""
        if scope_model.is_restricted():
            scope_note = f"\n用户标注范围：{scope_model.scope_summary or scope_model.to_payload().model_dump()}"

        normalized = _coords_are_normalized(normalized_boxes)
        regions = _crop_boxes_from_image_bytes(
            image_bytes,
            normalized_boxes,
            normalized=normalized,
        )
        mappings = await _vision_map_regions_concurrent(
            llm,
            regions,
            candidates=candidates,
            user_request=user_request,
            intent_summary=intent_summary,
            scope_note=scope_note,
            concurrency=concurrency,
        )

        unmapped = [m["box_index"] for m in mappings if not m.get("label_id")]
        mapped_n = len(mappings) - len(unmapped)
        log_annotation_agent(
            "map-done",
            "vision_crop",
            mapped=mapped_n,
            unmapped=len(unmapped),
            crop_regions=len(regions),
            image_source=image_source,
            vision_map_concurrency=concurrency,
        )
        return {
            "ok": True,
            "method": "vision_crop",
            "mappings": mappings,
            "unmapped_indices": unmapped,
            "label_candidates": candidates,
            "next_step": "finalize_image_change",
        }

    heuristic_raw = heuristic_map_boxes(
        [
            {
                "box_index": b["box_index"],
                "class_name": b.get("class_name") or b.get("detection_label"),
                "confidence": b.get("confidence"),
            }
            for b in normalized_boxes
        ],
        candidates,
        ocr_text=ocr_text,
    )
    mappings = [
        {
            "box_index": int(m.get("box_index", 0)),
            "label_id": str(m.get("label_id") or ""),
            "reason": m.get("reason") or "",
        }
        for m in heuristic_raw
    ]
    unmapped = [m["box_index"] for m in mappings if not m.get("label_id")]
    mapped_n = len(mappings) - len(unmapped)
    hint = ""
    if unmapped:
        hint = (
            "当前未启用视觉模型或未提供图像，多人物/实例标签无法可靠区分；"
            "请使用通过视觉探针的多模态模型，并确保 use_vision_mapping=true"
        )
    log_annotation_agent(
        "map-done",
        "heuristic",
        mapped=mapped_n,
        unmapped=len(unmapped),
        hint=hint or None,
        sample_detection_classes=[
            str(b.get("class_name") or b.get("detection_label") or "") for b in normalized_boxes[:6]
        ],
        sample_label_names=[str(c.get("name") or "") for c in candidates[:12]],
    )
    return {
        "ok": len(unmapped) < len(mappings),
        "method": "heuristic",
        "mappings": mappings,
        "unmapped_indices": unmapped,
        "label_candidates": candidates,
        "hint": hint,
        "next_step": "finalize_image_change",
    }
