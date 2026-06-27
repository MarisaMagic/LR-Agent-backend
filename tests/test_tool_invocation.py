import json

from app.agent.tool_invocation import (
    parse_pseudo_tool_calls,
    resolve_tool_calls,
)


def test_parse_pseudo_tool_calls_from_markdown() -> None:
    user = "帮我标注 data 文件夹下的所有图片"
    text = '我将调用 `execute_batch_annotation(user_request="data 文件夹")` 开始标注。'
    calls = parse_pseudo_tool_calls(text, user)
    assert len(calls) == 1
    assert calls[0].name == "execute_batch_annotation"
    assert calls[0].source == "parsed"
    assert user in calls[0].arguments["user_request"]


def test_parse_pseudo_tool_calls_skips_completed() -> None:
    text = "execute_batch_annotation(user_request='x')"
    calls = parse_pseudo_tool_calls(
        text,
        "标注并写报告",
        completed_tools=frozenset({"execute_batch_annotation"}),
    )
    assert calls == []


def test_resolve_api_tool_calls_first() -> None:
    api = [{"id": "t1", "name": "analyze_data", "args": {"user_request": "分析"}}]
    resolved = resolve_tool_calls(
        api_tool_calls=api,
        response_text="execute_batch_annotation(...)",
        user_content="分析数据",
    )
    assert len(resolved) == 1
    assert resolved[0].name == "analyze_data"
    assert resolved[0].source == "api"


def test_resolve_skips_completed_api_calls() -> None:
    api = [{"id": "t1", "name": "execute_batch_annotation", "args": {}}]
    resolved = resolve_tool_calls(
        api_tool_calls=api,
        response_text="",
        user_content="标注并写报告",
        completed_tools=frozenset({"execute_batch_annotation"}),
    )
    assert resolved == []


def test_resolve_pseudo_when_no_api() -> None:
    user = "批量标注所有图片"
    text = "execute_batch_annotation(user_request='所有图片')"
    resolved = resolve_tool_calls(
        api_tool_calls=[],
        response_text=text,
        user_content=user,
    )
    assert len(resolved) == 1
    assert resolved[0].source == "parsed"


def test_resolve_no_deterministic_from_user_message() -> None:
    """用户消息含「文件夹下」等短语时，不得自动派发客户端工具。"""
    user = "写一份数据报告，放在 /reports 文件夹下。"
    resolved = resolve_tool_calls(
        api_tool_calls=[],
        response_text="好的，我来处理。",
        user_content=user,
    )
    assert resolved == []


def test_resolve_write_file_task_no_batch_dispatch() -> None:
    user = "写一个 dijkstra 算法，保存到 src/dijkstra.cpp"
    resolved = resolve_tool_calls(
        api_tool_calls=[],
        response_text="我将为您编写代码。",
        user_content=user,
    )
    assert resolved == []


def test_parsed_args_user_request_in_result_json_shape() -> None:
    calls = parse_pseudo_tool_calls(
        'execute_batch_annotation(user_request="短")',
        "帮我标注 data 文件夹下的所有图片",
    )
    assert calls[0].arguments["user_request"] == "帮我标注 data 文件夹下的所有图片"
    assert json.dumps(calls[0].arguments)
