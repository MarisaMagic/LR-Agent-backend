"""Vision capability from DB probe results (no model-name keyword matching)."""
from __future__ import annotations

from typing import Any


def provider_supports_vision(row: Any) -> bool:
    """True when API vision probe succeeded for this provider row."""
    return bool(getattr(row, "supports_vision", False))


def is_vision_provider_row(row: Any) -> bool:
    return provider_supports_vision(row)


# Backward-compatible alias; keyword list removed — use llm_vision_probe + DB field.
def is_vision_provider(*, model: str = "", name: str = "") -> bool:
    del model, name
    return False
