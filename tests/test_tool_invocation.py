import json

from app.agent.tool_invocation import (
    normalize_api_tool_calls,
)


def test_normalize_api_tool_calls_standard() -> None:
    api = [{"id": "t1", "name": "analyze_data", "args": {"user_request": "分析"}}]
    resolved = normalize_api_tool_calls(api_tool_calls=api)
    assert len(resolved) == 1
    assert resolved[0].name == "analyze_data"
    assert resolved[0].source == "api"


def test_normalize_skips_completed_api_calls() -> None:
    api = [{"id": "t1", "name": "auto_annotate", "args": {}}]
    resolved = normalize_api_tool_calls(
        api_tool_calls=api,
        completed_tools=frozenset({"auto_annotate"}),
    )
    assert resolved == []


def test_normalize_empty_api_calls() -> None:
    resolved = normalize_api_tool_calls(api_tool_calls=[])
    assert resolved == []


def test_normalize_with_string_args_coerces_to_dict() -> None:
    api = [
        {"id": "t1", "name": "read_workspace_file", "args": json.dumps({"relative_path": "test.py"})}
    ]
    resolved = normalize_api_tool_calls(api_tool_calls=api)
    assert len(resolved) == 1
    assert resolved[0].arguments["relative_path"] == "test.py"


def test_normalize_skips_empty_name() -> None:
    api = [{"id": "t1", "name": "", "args": {}}]
    resolved = normalize_api_tool_calls(api_tool_calls=api)
    assert resolved == []
