from app.agent.analysis_prepare_service import (
    _build_human_message,
    _normalize_analysis_script,
    _validate_analysis_script,
)
from app.agent.analysis_prepare_service import AnalysisRepairContext


def test_normalize_dedent_over_indented_script():
    raw = "    import json\n    data = DATA\n    print(data['totalBoxes'])"
    normalized = _normalize_analysis_script(raw)
    assert normalized.startswith("import json")
    assert _validate_analysis_script(normalized) is None


def test_validate_rejects_open():
    script = "with open('x') as f:\n    pass"
    assert _validate_analysis_script(script) is not None


def test_build_human_includes_repair_context():
    human = _build_human_message(
        user_request="统计标签",
        data_snapshot={"labelCounts": {"a": 1}},
        conversation_transcript="",
        repair_context=AnalysisRepairContext(
            previous_script="bad script",
            error_message="语法错误: unexpected indent",
            error_stage="syntax",
        ),
    )
    assert "【上次脚本】" in human
    assert "unexpected indent" in human

