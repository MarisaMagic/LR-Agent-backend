"""Structured logs for batch annotation debugging (uvicorn terminal)."""
from __future__ import annotations

import json
import logging
from typing import Any

logger = logging.getLogger(__name__)


def log_annotation_agent(stage: str, message: str, **fields: Any) -> None:
    if fields:
        try:
            extra = json.dumps(fields, ensure_ascii=False, default=str)
        except (TypeError, ValueError):
            extra = str(fields)
        logger.info("[annotation_agent] [%s] %s | %s", stage, message, extra)
    else:
        logger.info("[annotation_agent] [%s] %s", stage, message)
