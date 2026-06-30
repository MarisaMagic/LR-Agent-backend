"""测试 AssistModeRouter：turn_kind → mode 映射表穷举验证。"""

import pytest
from app.agent.assist_mode_router import (
    AssistMode,
    AssistModeRouter,
    FULL_TOOL_SET,
    LIGHT_TOOL_SET,
    TURN_KIND_MODE_MAP,
)


def test_converse_routes_to_light_with_workspace():
    mode, tool_set = AssistModeRouter.resolve(
        "converse", has_project_snapshot=False, has_workspace=True
    )
    assert mode == AssistMode.LIGHT
    assert "write_workspace_file" in tool_set
    assert "execute_batch_annotation" not in tool_set


def test_execute_batch_routes_to_full_with_project():
    mode, tool_set = AssistModeRouter.resolve(
        "execute_batch", has_project_snapshot=True, has_workspace=True
    )
    assert mode == AssistMode.FULL
    assert "execute_batch_annotation" in tool_set


def test_full_downgrades_to_light_without_workspace():
    mode, tool_set = AssistModeRouter.resolve(
        "execute_batch", has_project_snapshot=False, has_workspace=False
    )
    assert mode == AssistMode.LIGHT
    assert "execute_batch_annotation" not in tool_set


def test_clarify_scope_routes_to_chat():
    mode, tool_set = AssistModeRouter.resolve(
        "clarify_scope", has_project_snapshot=True, has_workspace=True
    )
    assert mode == AssistMode.CHAT
    assert len(tool_set) == 0


def test_unsupported_routes_to_chat():
    mode, tool_set = AssistModeRouter.resolve(
        "unsupported", has_project_snapshot=True, has_workspace=True
    )
    assert mode == AssistMode.CHAT


def test_all_mapped_kinds_have_entries():
    """确保 TURN_KIND_MODE_MAP 覆盖了所有使用的 turn_kind 值。"""
    expected_kinds = {
        "converse", "generate_report", "generate_document",
        "execute_batch", "mutate_annotation", "analyze_data",
        "clarify_scope", "unsupported", "wants_batch", "query_annotation",
    }
    assert set(TURN_KIND_MODE_MAP.keys()) == expected_kinds


def test_generate_report_has_write_tool():
    mode, tool_set = AssistModeRouter.resolve(
        "generate_report", has_workspace=True
    )
    assert "write_workspace_file" in tool_set


def test_light_mode_does_not_include_async_tools():
    mode, tool_set = AssistModeRouter.resolve(
        "converse", has_workspace=True
    )
    assert "execute_batch_annotation" not in tool_set
    assert "mutate_annotation" not in tool_set
    assert "analyze_data" not in tool_set
