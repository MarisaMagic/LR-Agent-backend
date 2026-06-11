"""复用 LR-Agent-analysis 的 script_guard 做 prepare 阶段预检。"""

from __future__ import annotations

import sys
from pathlib import Path

_ANALYSIS_ROOT = Path(__file__).resolve().parents[3] / "LR-Agent-analysis"
if str(_ANALYSIS_ROOT) not in sys.path:
    sys.path.insert(0, str(_ANALYSIS_ROOT))

from script_guard import ScriptGuardError, validate_script  # noqa: E402

__all__ = ["ScriptGuardError", "validate_script"]
