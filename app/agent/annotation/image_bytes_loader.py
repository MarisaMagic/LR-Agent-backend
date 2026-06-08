"""图像字节加载：从本地绝对路径或 base64 获取原始图像数据。

Electron 客户端与后端 API 同机部署时优先读 image_absolute_path；
否则回退 image_base64。供 map_labels_service 裁剪、chat_message_builder 多模态消息使用。
"""

from __future__ import annotations

import base64
from pathlib import Path


def load_image_bytes(
    *,
    image_absolute_path: str = "",
    image_base64: str = "",
) -> tuple[bytes | None, str]:
    """加载图像字节，返回 (bytes, source)，source 为 path | base64 | none。"""
    path_str = (image_absolute_path or "").strip()
    if path_str:
        try:
            path = Path(path_str)
            if path.is_file():
                return path.read_bytes(), "path"
        except OSError:
            pass

    raw = (image_base64 or "").strip()
    if not raw:
        return None, "none"
    try:
        if "," in raw[:80]:
            raw = raw.split(",", 1)[1]
        return base64.b64decode(raw), "base64"
    except Exception:
        return None, "none"
