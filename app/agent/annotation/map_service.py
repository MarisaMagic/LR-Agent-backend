from __future__ import annotations

import json

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

from app.agent.annotation.json_utils import extract_json_object
from app.agent.annotation.schemas import BoxMappingItem, MapBoxesResult

MAP_SYSTEM = """你是检测框到任务标签的映射助手。
根据用户标注意图、标签树候选，为每个检测框推荐 label_id。
只能使用候选中的 label_id；无法确定时可跳过该框。

检测类名（YOLO class）与任务标签名（label_tree）可能不一致，需结合用户意图与图像理解映射。

输出 JSON：{"mappings":[{"box_index":0,"label_id":"..."},...]}"""

VISION_MAP_SYSTEM = """你是视觉标注助手。你会收到一张图片（base64）和若干检测框。
请根据图像内容与用户意图，为每个框选择最合适的 label_id（只能使用标签候选中的 id）。

输出 JSON：{"mappings":[{"box_index":0,"label_id":"..."},...]}"""

CROP_VISION_SYSTEM = """你是视觉标注助手。你会收到一张裁剪后的目标区域图。
请根据图像内容与用户标注意图，从标签候选中选择最合适的 label_id。
若裁剪区域与任一候选标签语义不符，必须返回空 label_id。
只输出 JSON：{"label_id":"...","reason":"..."} ；无法确定或与任务无关时 label_id 必须为 ""。"""


async def map_detection_boxes(
    llm: ChatOpenAI,
    *,
    user_request: str,
    intent_summary: str,
    label_candidates: list[dict],
    boxes: list[dict],
    label_strategy: str,
    single_label_id: str | None = None,
) -> MapBoxesResult:
    if label_strategy == "single_label_for_all_boxes" and single_label_id:
        mappings = [
            BoxMappingItem(box_index=i, label_id=single_label_id)
            for i in range(len(boxes))
        ]
        return MapBoxesResult(mappings=mappings)

    messages = [
        SystemMessage(content=MAP_SYSTEM),
        HumanMessage(
            content=(
                f"用户请求：{user_request}\n\n"
                f"意图：{intent_summary}\n\n"
                f"标签候选：{json.dumps(label_candidates[:80], ensure_ascii=False)}\n\n"
                f"检测框：\n{json.dumps(boxes[:40], ensure_ascii=False)}"
            )
        ),
    ]
    resp = await llm.ainvoke(messages)
    content = resp.content if hasattr(resp, "content") else str(resp)
    data = extract_json_object(str(content))
    raw = data.get("mappings")
    mappings: list[BoxMappingItem] = []
    if isinstance(raw, list):
        for item in raw:
            if not isinstance(item, dict):
                continue
            try:
                mappings.append(
                    BoxMappingItem(
                        box_index=int(item.get("box_index", 0)),
                        label_id=str(item.get("label_id") or ""),
                    )
                )
            except (TypeError, ValueError):
                continue
    return MapBoxesResult(mappings=mappings)


async def map_detection_boxes_with_vision(
    llm: ChatOpenAI,
    *,
    user_request: str,
    intent_summary: str,
    label_candidates: list[dict],
    boxes: list[dict],
    image_base64: str,
    mime_type: str = "image/jpeg",
) -> MapBoxesResult:
    if not image_base64.strip():
        return await map_detection_boxes(
            llm,
            user_request=user_request,
            intent_summary=intent_summary,
            label_candidates=label_candidates,
            boxes=boxes,
            label_strategy="map_each_box_to_label",
        )

    human_content: list[dict] = [
        {
            "type": "text",
            "text": (
                f"用户请求：{user_request}\n意图：{intent_summary}\n"
                f"标签候选：{json.dumps(label_candidates[:80], ensure_ascii=False)}\n"
                f"检测框：{json.dumps(boxes[:40], ensure_ascii=False)}"
            ),
        },
        {
            "type": "image_url",
            "image_url": {"url": f"data:{mime_type};base64,{image_base64}"},
        },
    ]
    messages = [
        SystemMessage(content=VISION_MAP_SYSTEM),
        HumanMessage(content=human_content),
    ]
    try:
        resp = await llm.ainvoke(messages)
    except Exception:
        return await map_detection_boxes(
            llm,
            user_request=user_request,
            intent_summary=intent_summary,
            label_candidates=label_candidates,
            boxes=boxes,
            label_strategy="map_each_box_to_label",
        )
    content = resp.content if hasattr(resp, "content") else str(resp)
    data = extract_json_object(str(content))
    raw = data.get("mappings")
    mappings: list[BoxMappingItem] = []
    if isinstance(raw, list):
        for item in raw:
            if not isinstance(item, dict):
                continue
            try:
                mappings.append(
                    BoxMappingItem(
                        box_index=int(item.get("box_index", 0)),
                        label_id=str(item.get("label_id") or ""),
                    )
                )
            except (TypeError, ValueError):
                continue
    return MapBoxesResult(mappings=mappings)


async def map_single_box_crop_vision(
    llm: ChatOpenAI,
    *,
    user_request: str,
    intent_summary: str,
    label_candidates: list[dict],
    box_index: int,
    class_name: str,
    crop_base64: str,
    mime_type: str = "image/jpeg",
) -> BoxMappingItem:
    if not crop_base64.strip():
        return BoxMappingItem(box_index=box_index, label_id="")

    human_content: list[dict] = [
        {
            "type": "text",
            "text": (
                f"用户请求：{user_request}\n意图：{intent_summary}\n"
                f"检测类名：{class_name}\n"
                f"box_index：{box_index}\n"
                f"标签候选：{json.dumps(label_candidates[:40], ensure_ascii=False)}"
            ),
        },
        {
            "type": "image_url",
            "image_url": {"url": f"data:{mime_type};base64,{crop_base64}"},
        },
    ]
    messages = [
        SystemMessage(content=CROP_VISION_SYSTEM),
        HumanMessage(content=human_content),
    ]
    try:
        resp = await llm.ainvoke(messages)
    except Exception:
        return BoxMappingItem(box_index=box_index, label_id="")
    content = resp.content if hasattr(resp, "content") else str(resp)
    data = extract_json_object(str(content))
    label_id = str(data.get("label_id") or "").strip()
    valid = {str(c.get("id") or "") for c in label_candidates}
    if label_id and label_id not in valid:
        label_id = ""
    return BoxMappingItem(box_index=box_index, label_id=label_id)
