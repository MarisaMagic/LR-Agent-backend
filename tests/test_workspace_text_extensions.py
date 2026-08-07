from app.agent.tools.workspace_text_extensions import (
    is_allowed_text_extension,
    is_blocked_text_extension,
)


def test_jsonl_allowed():
    assert not is_blocked_text_extension(".jsonl")
    assert is_allowed_text_extension(".jsonl")


def test_png_blocked():
    assert is_blocked_text_extension(".png")
    assert not is_allowed_text_extension(".png")


def test_extensionless_allowed():
    assert not is_blocked_text_extension("")
    assert is_allowed_text_extension("")
