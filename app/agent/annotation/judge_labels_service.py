"""Whole-image judge for detection-box label assignments."""

from __future__ import annotations

import json
from typing import Any, Literal

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

from app.agent.annotation.image_bytes_loader import image_bytes_to_data_url, load_image_bytes
from app.agent.annotation.json_utils import extract_json_object
from app.agent.annotation.debug_log import log_annotation_agent
from app.core.config import Settings, get_settings

JudgeVerdict = Literal["accept", "weak_accept", "reject"]

JUDGE_SYSTEM = """你是标注质量评分子 Agent（JudgeAgent）。
你会收到整张图片、检测框、框到标签的映射、最终标注草案和标签候选。
请重点判断是否存在标签错误、框与标签不一致、漏掉明显应标目标、或把无关目标误标为候选标签。

返回规则：
- accept：标签基本正确，可直接通过。
- weak_accept：整体可用但存在轻微不确定性，需要用户界面提示人工复查。
- reject：存在明确标签错误或关键不确定性，需要打标签子 Agent 结合反馈重做整张图映射。

只输出 JSON：
{
  "verdict": "accept" | "weak_accept" | "reject",
  "confidence": 0.0,
  "summary": "...",
  "issues": [{"box_index": 0, "code": "...", "message": "...", "expected_label_id": "", "actual_label_id": ""}],
  "retry_feedback": "...",
  "checked_boxes": 0
}
若 reject，retry_feedback 必须说明打标签子 Agent 下一轮应如何修正。"""


def _normalize_verdict(value: Any) -> JudgeVerdict:
    raw = str(value or "").strip().lower().replace("-", "_")
    if raw in {"accept", "weak_accept", "reject"}:
        return raw  # type: ignore[return-value]
    return "weak_accept"


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    return max(0.0, min(1.0, out))


def _normalize_issues(raw: Any) -> list[dict[str, Any]]:
    if not isinstance(raw, list):
        return []
    issues: list[dict[str, Any]] = []
    for item in raw[:20]:
        if not isinstance(item, dict):
            continue
        message = str(item.get("message") or "").strip()
        if not message:
            continue
        issue: dict[str, Any] = {
            "message": message,
        }
        if item.get("box_index") is not None:
            try:
                issue["box_index"] = int(item.get("box_index"))
            except (TypeError, ValueError):
                pass
        for src, dst in (
            ("code", "code"),
            ("expected_label_id", "expected_label_id"),
            ("actual_label_id", "actual_label_id"),
        ):
            value = str(item.get(src) or "").strip()
            if value:
                issue[dst] = value
        issues.append(issue)
    return issues


async def judge_detection_labels(
    llm: ChatOpenAI,
    *,
    user_request: str,
    intent_summary: str,
    label_candidates: list[dict],
    boxes: list[dict],
    mappings: list[dict],
    annotations: list[dict],
    image_absolute_path: str = "",
    image_base64: str = "",
    attempt: int = 0,
    max_retries: int = 3,
    settings: Settings | None = None,
) -> dict[str, Any]:
    """Judge a whole-image annotation draft with the original image context."""

    cfg = settings or get_settings()
    image_bytes, image_source = load_image_bytes(
        image_absolute_path=image_absolute_path,
        image_base64=image_base64,
    )
    if image_bytes is None:
        return {
            "ok": False,
            "error": "image_unavailable",
            "verdict": "reject",
            "confidence": 0.0,
            "summary": "评分需要可读整图",
            "issues": [],
            "retry_feedback": "请提供可读整图后再进行评分。",
            "checked_boxes": 0,
        }

    label_payload = [
        {"id": c.get("id"), "name": c.get("name")}
        for c in label_candidates[:80]
    ]
    user_text = (
        f"用户请求：{user_request}\n"
        f"意图：{intent_summary}\n"
        f"当前评分轮次：{attempt + 1} / {max_retries + 1}\n\n"
        f"标签候选：\n{json.dumps(label_payload, ensure_ascii=False)}\n\n"
        f"检测框：\n{json.dumps(boxes[:80], ensure_ascii=False)}\n\n"
        f"框到标签映射：\n{json.dumps(mappings[:80], ensure_ascii=False)}\n\n"
        f"最终标注草案：\n{json.dumps(annotations[:80], ensure_ascii=False)}"
    )
    data_url = image_bytes_to_data_url(
        image_bytes,
        max_edge=cfg.agent_chat_vision_max_edge,
        jpeg_quality=cfg.agent_chat_vision_jpeg_quality,
    )
    messages = [
        SystemMessage(content=JUDGE_SYSTEM),
        HumanMessage(
            content=[
                {"type": "text", "text": user_text},
                {"type": "image_url", "image_url": {"url": data_url}},
            ]
        ),
    ]

    resp = await llm.ainvoke(messages)
    content = resp.content if hasattr(resp, "content") else str(resp)
    if isinstance(content, list):
        content = "".join(
            part.get("text", "") if isinstance(part, dict) else str(part)
            for part in content
        )
    parsed = extract_json_object(str(content))
    verdict = _normalize_verdict(parsed.get("verdict"))
    confidence = _safe_float(parsed.get("confidence"), 0.0)
    issues = _normalize_issues(parsed.get("issues"))
    summary = str(parsed.get("summary") or "").strip()
    retry_feedback = str(parsed.get("retry_feedback") or parsed.get("retryFeedback") or "").strip()
    if verdict == "reject" and not retry_feedback:
        retry_feedback = summary or "评分子 Agent 判定存在标签错误，请重新结合整图和检测框分配标签。"
    try:
        checked_boxes = int(parsed.get("checked_boxes") or parsed.get("checkedBoxes") or len(boxes))
    except (TypeError, ValueError):
        checked_boxes = len(boxes)

    log_annotation_agent(
        "judge-done",
        "JudgeAgent 评分完成",
        verdict=verdict,
        confidence=confidence,
        issue_count=len(issues),
        checked_boxes=checked_boxes,
        attempt=attempt,
        image_source=image_source,
    )

    return {
        "ok": True,
        "verdict": verdict,
        "confidence": confidence,
        "summary": summary,
        "issues": issues,
        "retry_feedback": retry_feedback,
        "checked_boxes": checked_boxes,
    }
