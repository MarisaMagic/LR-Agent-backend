"""工具注册元数据：runner 类型、能力分类与副作用分类（Agent 调度单一真相源）。"""

from __future__ import annotations

from enum import Enum


class ToolRunner(str, Enum):
    """工具执行器类型。"""

    SYNC = "sync"  # 本轮在后端同步执行完毕
    PROPOSAL = "proposal"  # 后端校验并生成提案 SSE，落盘由 Electron 确认
    ASYNC = "async"  # 暂停 loop，由 Electron 执行后 resume


class ToolCapability(str, Enum):
    """工具能力抽象——每个经典工具对应一个唯一能力，防止双轨制。"""

    READ_TEXT_FILE = "read_text_file"
    READ_DOCUMENT = "read_document"
    READ_IMAGE_VISION = "read_image_vision"
    READ_ANNOTATION_JSON = "read_annotation_json"
    WRITE_FILE = "write_file"
    DETECT_BATCH = "detect_batch"
    MUTATE_ANNOTATION = "mutate_annotation_cap"
    ANALYZE_DATA = "analyze_data_cap"
    QUERY_CONTEXT = "query_context"


# 内置工具 → 唯一能力映射
TOOL_CAPABILITY_MAP: dict[str, ToolCapability] = {
    "read_workspace_file": ToolCapability.READ_TEXT_FILE,
    "read_document_file": ToolCapability.READ_DOCUMENT,
    "read_image_for_vision": ToolCapability.READ_IMAGE_VISION,
    "read_file_annotation": ToolCapability.READ_ANNOTATION_JSON,
    "write_workspace_file": ToolCapability.WRITE_FILE,
    "execute_batch_annotation": ToolCapability.DETECT_BATCH,
    "mutate_annotation": ToolCapability.MUTATE_ANNOTATION,
    "analyze_data": ToolCapability.ANALYZE_DATA,
    "get_account_summary": ToolCapability.QUERY_CONTEXT,
    "get_lr_agent_help": ToolCapability.QUERY_CONTEXT,
    "describe_client_context": ToolCapability.QUERY_CONTEXT,
    "describe_annotation_project": ToolCapability.QUERY_CONTEXT,
}

# 所有内置工具的能力集合
CANONICAL_CAPABILITIES: frozenset[ToolCapability] = frozenset(TOOL_CAPABILITY_MAP.values())


# 内置工具 runner 映射（MCP 动态工具默认 SYNC，且不与下列重名注入）
TOOL_RUNNERS: dict[str, ToolRunner] = {
    "get_account_summary": ToolRunner.SYNC,
    "get_lr_agent_help": ToolRunner.SYNC,
    "describe_client_context": ToolRunner.SYNC,
    "describe_annotation_project": ToolRunner.SYNC,
    "read_file_annotation": ToolRunner.SYNC,
    "read_workspace_file": ToolRunner.SYNC,
    "read_image_for_vision": ToolRunner.SYNC,
    "read_document_file": ToolRunner.SYNC,
    "write_workspace_file": ToolRunner.PROPOSAL,
    "execute_batch_annotation": ToolRunner.ASYNC,
    "mutate_annotation": ToolRunner.ASYNC,
    "analyze_data": ToolRunner.ASYNC,
}

# 不与 MCP 重复暴露的本地实现工具名
LOCAL_CANONICAL_TOOL_NAMES: frozenset[str] = frozenset(TOOL_RUNNERS.keys())

ASYNC_TOOL_NAMES: frozenset[str] = frozenset(
    name for name, runner in TOOL_RUNNERS.items() if runner is ToolRunner.ASYNC
)

# 向后兼容别名
CLIENT_TOOL_NAMES = ASYNC_TOOL_NAMES


def get_tool_runner(name: str) -> ToolRunner:
    return TOOL_RUNNERS.get(name, ToolRunner.SYNC)


def is_async_tool(name: str) -> bool:
    return get_tool_runner(name) is ToolRunner.ASYNC


def is_proposal_tool(name: str) -> bool:
    return get_tool_runner(name) is ToolRunner.PROPOSAL
