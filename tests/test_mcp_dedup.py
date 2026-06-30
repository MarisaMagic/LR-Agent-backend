"""测试 MCP 工具注入时的 capability 去重。"""

import pytest
from app.agent.tools.tool_registry_meta import (
    CANONICAL_CAPABILITIES,
    TOOL_CAPABILITY_MAP,
    ToolCapability,
)
from app.agent.tools.mcp_client import _infer_mcp_capability


def test_infer_write_workspace_file():
    cap = _infer_mcp_capability("write_workspace_file")
    assert cap == ToolCapability.WRITE_FILE


def test_infer_yolo_detect():
    cap = _infer_mcp_capability("yolo_detect")
    assert cap == ToolCapability.DETECT_BATCH


def test_infer_list_project_images():
    cap = _infer_mcp_capability("list_project_images")
    assert cap == ToolCapability.QUERY_CONTEXT


def test_infer_unknown_tool():
    cap = _infer_mcp_capability("some_unknown_tool")
    assert cap is None


def test_canonical_capabilities_do_not_overlap_mcp_unique():
    """确保 MCP 注入的 yolo_detect 会被过滤（能力冲突）。"""
    cap = _infer_mcp_capability("yolo_detect")
    assert cap is not None
    assert cap in CANONICAL_CAPABILITIES  # 存在 canonical 实现，应当 filtering


def test_all_canonical_tools_have_capability():
    from app.agent.tools.tool_registry_meta import TOOL_RUNNERS
    for name in TOOL_RUNNERS:
        assert name in TOOL_CAPABILITY_MAP, f"工具 {name} 缺少能力映射"
