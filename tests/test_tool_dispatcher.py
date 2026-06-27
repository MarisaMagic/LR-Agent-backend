import json

from app.agent.tool_dispatcher import resolve_round_tool_calls, split_resolved_calls
from app.agent.tool_invocation import (
    _parse_write_workspace_file_pseudo,
    parse_pseudo_tool_calls,
    resolve_tool_calls,
    tool_mentioned_in_text,
)
from app.agent.tools.tool_registry_meta import ToolRunner, get_tool_runner


def test_get_tool_runner_write_is_proposal() -> None:
    assert get_tool_runner("write_workspace_file") is ToolRunner.PROPOSAL


def test_get_tool_runner_batch_is_async() -> None:
    assert get_tool_runner("execute_batch_annotation") is ToolRunner.ASYNC


def test_parse_write_workspace_file_pseudo() -> None:
    text = '''
    write_workspace_file(
        relative_path="dijkstra_algorithm/dijkstra.cpp",
        content=\"\"\"
#include <iostream>
int main() { return 0; }
\"\"\"
    )
    '''
    args = _parse_write_workspace_file_pseudo(text)
    assert args is not None
    assert args["relative_path"] == "dijkstra_algorithm/dijkstra.cpp"
    assert "#include" in args["content"]


def test_resolve_parsed_write_workspace_file() -> None:
    text = 'write_workspace_file(relative_path="reports/a.md", content="hello")'
    calls = resolve_tool_calls(
        api_tool_calls=[],
        response_text=text,
        user_content="写报告",
    )
    assert len(calls) == 1
    assert calls[0].name == "write_workspace_file"
    assert calls[0].source == "parsed"


def test_split_immediate_and_async() -> None:
    from app.agent.tool_invocation import ResolvedToolCall

    calls = [
        ResolvedToolCall("t1", "write_workspace_file", {"relative_path": "a.md", "content": "x"}, "api"),
        ResolvedToolCall("t2", "execute_batch_annotation", {"user_request": "标注"}, "api"),
    ]
    split = split_resolved_calls(calls)
    assert len(split.immediate) == 1
    assert split.immediate[0].name == "write_workspace_file"
    assert len(split.async_pending) == 1
    assert split.async_pending[0].name == "execute_batch_annotation"


def test_tool_mentioned_in_text_write() -> None:
    assert tool_mentioned_in_text('call write_workspace_file(path="a")') == "write_workspace_file"


def test_build_tool_result_shape() -> None:
    from app.agent.tools.tool_result import build_tool_result, parse_tool_result

    raw = build_tool_result(
        ok=True,
        tool="write_workspace_file",
        status="proposal_ready",
        summary="提案已生成",
        proposal_pending=True,
    )
    data = parse_tool_result(raw)
    assert data is not None
    assert data["ok"] is True
    assert data["proposal_pending"] is True
    assert json.loads(raw)["tool"] == "write_workspace_file"
