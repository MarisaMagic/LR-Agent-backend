"""Unit tests for vision mapping validation."""

from app.agent.annotation.map_validation_service import (
    MappingIssueCode,
    candidates_for_retry_box,
    find_label_names_in_text,
    validate_vision_mappings,
)

CANDIDATES = [
    {"id": "a", "name": "curry"},
    {"id": "b", "name": "james"},
    {"id": "c", "name": "durant"},
    {"id": "d", "name": "jokic"},
]


def test_reason_name_mismatch_detects_wrong_id():
    mappings = [
        {
            "box_index": 1,
            "label_id": "d",
            "reason": "图像中的人物是凯文·杜兰特（Kevin Durant）",
        }
    ]
    result = validate_vision_mappings(
        mappings,
        CANDIDATES,
        instance_labels=True,
    )
    codes = [i.code for i in result.issues]
    assert MappingIssueCode.REASON_NAME_MISMATCH in codes


def test_duplicate_label_detected():
    mappings = [
        {"box_index": 0, "label_id": "a", "reason": "curry"},
        {"box_index": 1, "label_id": "a", "reason": "also curry"},
    ]
    result = validate_vision_mappings(mappings, CANDIDATES, instance_labels=True)
    assert any(i.code == MappingIssueCode.DUPLICATE_LABEL for i in result.issues)


def test_valid_mapping_passes():
    mappings = [
        {"box_index": 0, "label_id": "b", "reason": "勒布朗·詹姆斯"},
        {"box_index": 1, "label_id": "c", "reason": "凯文·杜兰特"},
    ]
    result = validate_vision_mappings(mappings, CANDIDATES, instance_labels=True)
    assert result.ok


def test_candidates_for_retry_excludes_used():
    filtered = candidates_for_retry_box(CANDIDATES, {"a", "b"}, keep_label_id="")
    ids = {c["id"] for c in filtered}
    assert "a" not in ids
    assert "b" not in ids
    assert "c" in ids


def test_find_label_names_in_text():
    hits = find_label_names_in_text("这是 durant 的比赛", CANDIDATES)
    assert hits == ["durant"]
