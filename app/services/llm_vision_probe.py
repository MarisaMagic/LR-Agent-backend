"""Probe LLM provider APIs for image (multimodal) input support."""
from __future__ import annotations

import base64
import io
import logging
from typing import Any

from langchain_core.messages import HumanMessage

from app.agent.llm_factory import build_chat_model
from app.models.llm_provider import LlmProvider

logger = logging.getLogger(__name__)

PROBE_PROMPT = "这是一张纯色测试图。请只回复一个大写字母 OK，不要其它内容。"


def _tiny_jpeg_data_url() -> str:
    from PIL import Image

    img = Image.new("RGB", (64, 64), color=(180, 60, 60))
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=85)
    b64 = base64.b64encode(buf.getvalue()).decode("ascii")
    return f"data:image/jpeg;base64,{b64}"


def _response_looks_ok(content: Any) -> bool:
    if content is None:
        return False
    if isinstance(content, list):
        text = "".join(
            part.get("text", "") if isinstance(part, dict) else str(part)
            for part in content
        )
    else:
        text = str(content)
    return "ok" in text.lower()


async def probe_vision_capability(provider: LlmProvider, api_key: str) -> tuple[bool, str]:
    """
    Send a minimal image+text request; success means the endpoint accepts vision input.
    Returns (supports_vision, detail_for_storage).
    """
    llm = build_chat_model(provider, api_key, streaming=False, temperature=0)
    data_url = _tiny_jpeg_data_url()
    messages = [
        HumanMessage(
            content=[
                {"type": "text", "text": PROBE_PROMPT},
                {"type": "image_url", "image_url": {"url": data_url}},
            ]
        )
    ]
    try:
        resp = await llm.ainvoke(messages)
        content = resp.content if hasattr(resp, "content") else str(resp)
        if _response_looks_ok(content):
            return True, "probe_ok"
        return False, f"probe_no_ok_response:{str(content)[:120]}"
    except Exception as exc:
        logger.info(
            "vision_probe_failed provider=%s model=%s err=%s",
            provider.id,
            provider.model,
            repr(exc)[:200],
        )
        return False, f"probe_error:{type(exc).__name__}:{str(exc)[:200]}"
