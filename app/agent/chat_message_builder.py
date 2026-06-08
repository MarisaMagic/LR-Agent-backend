"""Build LangChain messages including optional vision attachment for the latest user turn."""

from __future__ import annotations

import base64

from langchain_core.messages import HumanMessage

from app.agent.annotation.image_bytes_loader import load_image_bytes, resize_image_to_jpeg_bytes


def _resize_image_bytes(
    raw: bytes,
    *,
    max_edge: int,
    jpeg_quality: int,
) -> tuple[bytes, str]:
    return resize_image_to_jpeg_bytes(raw, max_edge=max_edge, jpeg_quality=jpeg_quality)


def build_multimodal_user_message(
    text: str,
    *,
    image_absolute_path: str = "",
    image_base64: str = "",
    max_edge: int = 1280,
    jpeg_quality: int = 85,
) -> HumanMessage:
    raw, _source = load_image_bytes(
        image_absolute_path=image_absolute_path,
        image_base64=image_base64,
    )
    if raw is None:
        return HumanMessage(content=text)

    try:
        jpeg_bytes, mime = _resize_image_bytes(
            raw,
            max_edge=max_edge,
            jpeg_quality=jpeg_quality,
        )
    except Exception:
        return HumanMessage(content=text)

    b64 = base64.b64encode(jpeg_bytes).decode("ascii")
    data_url = f"data:{mime};base64,{b64}"
    content: list[dict] = [
        {"type": "text", "text": text},
        {"type": "image_url", "image_url": {"url": data_url}},
    ]
    return HumanMessage(content=content)
