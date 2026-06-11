from app.agent.text_sanitize import sanitize_json_value, sanitize_unicode_text


def test_sanitize_unicode_text_removes_surrogates():
    dirty = "标签统计" + "\udcad" + "完成"
    cleaned = sanitize_unicode_text(dirty)
    assert "\udcad" not in cleaned
    assert "标签统计" in cleaned


def test_sanitize_json_value_recursive():
    dirty = {
        "labelCounts": {"a\uDCAF": 1},
        "files": [{"relativePath": "x\uDCAF.jpg", "labelCounts": {}}],
    }
    cleaned = sanitize_json_value(dirty)
    dumped = str(cleaned)
    assert "\udcaf" not in dumped.lower()
