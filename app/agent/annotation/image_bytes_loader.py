"""Load image bytes from local path (Electron + co-located API) or base64 fallback."""
from __future__ import annotations

import base64
from pathlib import Path


def load_image_bytes(
    *,
    image_absolute_path: str = "",
    image_base64: str = "",
) -> tuple[bytes | None, str]:
    """Return (bytes, source) where source is path | base64 | none."""
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
