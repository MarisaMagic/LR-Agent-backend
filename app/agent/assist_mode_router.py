"""Assist 模式路由器：基于客户端上下文决定运行模式与工具集。

三种模式:
  - FULL:  完整工具集（标注、分析、写文件、读文件、视觉）— 标注项目内
  - LIGHT: 只读工具 + write_workspace_file（报告、文档、问答）— 编辑器 / 工作区
  - CHAT:  纯对话，无工具（由 chat_service 处理，不经过 assist）
"""

from enum import Enum


class AssistMode(str, Enum):
    FULL = "full"
    LIGHT = "light"
    CHAT = "chat"


# 每种模式的工具白名单
LIGHT_TOOL_SET: frozenset[str] = frozenset({
    "read_workspace_file",
    "grep_workspace",
    "list_workspace_directory",
    "read_document_file",
    "read_image_for_vision",
    "write_workspace_file",
    "get_account_summary",
    "get_lr_agent_help",
    "describe_client_context",
    "describe_annotation_project",
    "read_file_annotation",
})

FULL_TOOL_SET: frozenset[str] = frozenset({
    "read_workspace_file",
    "grep_workspace",
    "list_workspace_directory",
    "read_document_file",
    "read_image_for_vision",
    "write_workspace_file",
    "execute_batch_annotation",
    "mutate_annotation",
    "analyze_data",
    "get_account_summary",
    "get_lr_agent_help",
    "describe_client_context",
    "describe_annotation_project",
    "read_file_annotation",
})
