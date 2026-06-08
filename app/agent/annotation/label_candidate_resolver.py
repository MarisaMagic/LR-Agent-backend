"""标签候选池解析：scope 显式范围 / 可选整图 preflight。"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Literal

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

from app.agent.annotation.annotation_scope import (
    AnnotationScope,
    filter_label_candidates_by_scope,
)
from app.agent.annotation.json_utils import extract_json_object
from app.agent.annotation.image_bytes_loader import image_bytes_to_data_url, load_image_bytes
from app.core.config import Settings, get_settings

PreflightMode = Literal["off", "auto", "always"]

PREFLIGHT_SYSTEM = """你是视觉标注准备助手。根据整图与用户意图，从标签候选中选出本图实际可能出现的 label_id。
不要猜测图中没有的目标。若无法可靠判断，present_label_ids 必须为空并设置 uncertain=true。
只输出 JSON：{"present_label_ids":["..."], "uncertain": false}"""


@dataclass
class LabelPoolResult:
    candidates: list[dict]
    source: str = "full"
    excluded_names: list[str] = field(default_factory=list)
    preflight_label_ids: list[str] | None = None


def _should_run_preflight(
    mode: PreflightMode,
    *,
    candidate_count: int,
    box_count: int,
    min_extra: int,
) -> bool:
    if mode == "off" or box_count <= 0:
        return False
    if mode == "always":
        return True
    extra = candidate_count - box_count
    return extra >= min_extra


def _image_bytes_to_data_url(image_bytes: bytes, settings: Settings) -> str:
    return image_bytes_to_data_url(
        image_bytes,
        max_edge=settings.agent_chat_vision_max_edge,
        jpeg_quality=settings.agent_chat_vision_jpeg_quality,
    )


async def preflight_label_pool(
    llm: ChatOpenAI,
    image_bytes: bytes,
    candidates: list[dict],
    *,
    box_count: int,
    user_request: str,
    intent_summary: str,
    settings: Settings | None = None,
) -> tuple[list[str], bool]:
    """整图 preflight：返回 (present_label_ids, uncertain)。"""
    cfg = settings or get_settings()
    valid_ids = {str(c.get("id") or "") for c in candidates}
    payload = [
        {"id": c.get("id"), "name": c.get("name")}
        for c in candidates[:80]
    ]
    user_text = (
        f"用户请求：{user_request}\n意图：{intent_summary}\n"
        f"检测框数量：{box_count}\n\n标签候选：\n"
        f"{json.dumps(payload, ensure_ascii=False)}"
    )
    data_url = _image_bytes_to_data_url(image_bytes, cfg)
    messages = [
        SystemMessage(content=PREFLIGHT_SYSTEM),
        HumanMessage(
            content=[
                {"type": "text", "text": user_text},
                {"type": "image_url", "image_url": {"url": data_url}},
            ]
        ),
    ]
    try:
        resp = await llm.ainvoke(messages)
    except Exception:
        return [], True

    content = resp.content if hasattr(resp, "content") else str(resp)
    if isinstance(content, list):
        content = "".join(
            p.get("text", "") if isinstance(p, dict) else str(p) for p in content
        )
    data = extract_json_object(str(content))
    uncertain = bool(data.get("uncertain", False))
    raw_ids = data.get("present_label_ids") or data.get("label_ids") or []
    if not isinstance(raw_ids, list):
        return [], True
    present = [str(i).strip() for i in raw_ids if str(i).strip() in valid_ids]
    if uncertain or not present:
        return [], True
    return present, False


async def resolve_effective_label_candidates(
    llm: ChatOpenAI | None,
    *,
    all_candidates: list[dict],
    scope: AnnotationScope,
    user_request: str,
    intent_summary: str,
    box_count: int,
    image_bytes: bytes | None = None,
    settings: Settings | None = None,
) -> LabelPoolResult:
    """L0 annotation_scope 显式标签范围 → L1 可选 preflight。

    user_request / intent_summary 不参与候选池缩小（避免 prepare 摘要举例误伤）。
    """
    cfg = settings or get_settings()
    scoped = filter_label_candidates_by_scope(list(all_candidates), scope)
    source = "scope" if scoped is not all_candidates and len(scoped) < len(all_candidates) else "full"
    excluded: list[str] = []

    preflight_ids: list[str] | None = None
    mode: PreflightMode = cfg.annotation_label_pool_preflight  # type: ignore[assignment]
    if (
        llm is not None
        and image_bytes
        and _should_run_preflight(
            mode,
            candidate_count=len(scoped),
            box_count=box_count,
            min_extra=cfg.annotation_label_pool_preflight_min_extra,
        )
    ):
        present, uncertain = await preflight_label_pool(
            llm,
            image_bytes,
            scoped,
            box_count=box_count,
            user_request=user_request,
            intent_summary=intent_summary,
            settings=cfg,
        )
        if not uncertain and present:
            id_set = set(present)
            pref_filtered = [c for c in scoped if str(c.get("id") or "") in id_set]
            if pref_filtered:
                preflight_ids = present
                excluded.extend(
                    str(c.get("name") or "")
                    for c in scoped
                    if str(c.get("id") or "") not in id_set
                )
                scoped = pref_filtered
                source = "preflight"

    return LabelPoolResult(
        candidates=scoped,
        source=source,
        excluded_names=[n for n in excluded if n],
        preflight_label_ids=preflight_ids,
    )
