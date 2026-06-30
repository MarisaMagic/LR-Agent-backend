"""Assist 模式路由器：根据 turn_kind 决定运行模式与工具集。

三种模式:
  - FULL:  完整工具集（标注、分析、写文件、读文件、视觉）
  - LIGHT: 只读工具 + write_workspace_file（报告、文档、问答）
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

# turn_kind → (AssistMode, 工具白名单)
TURN_KIND_MODE_MAP: dict[str, tuple[AssistMode, frozenset[str]]] = {
    "converse":             (AssistMode.LIGHT, LIGHT_TOOL_SET),
    "generate_report":      (AssistMode.LIGHT, LIGHT_TOOL_SET),
    "generate_document":    (AssistMode.LIGHT, LIGHT_TOOL_SET),
    "execute_batch":        (AssistMode.FULL,  FULL_TOOL_SET),
    "mutate_annotation":    (AssistMode.FULL,  FULL_TOOL_SET),
    "analyze_data":         (AssistMode.FULL,  FULL_TOOL_SET),
    "clarify_scope":        (AssistMode.CHAT,  frozenset()),
    "unsupported":          (AssistMode.CHAT,  frozenset()),
    # legacy / fallback kinds
    "wants_batch":          (AssistMode.FULL,  FULL_TOOL_SET),
    "query_annotation":     (AssistMode.LIGHT, LIGHT_TOOL_SET),
}


class AssistModeRouter:
    """根据 turn_kind 和客户端上下文决策运行模式。"""

    @staticmethod
    def resolve(
        turn_kind: str,
        *,
        has_project_snapshot: bool = False,
        has_workspace: bool = False,
    ) -> tuple[AssistMode, frozenset[str]]:
        """返回 (AssistMode, tool_set)。

        降级规则:
          - 无工作区时 FULL → LIGHT
          - 默认 turn_kind == "converse"
        """
        mode, tool_set = TURN_KIND_MODE_MAP.get(
            turn_kind, (AssistMode.LIGHT, LIGHT_TOOL_SET)
        )

        # 无工作区或项目时不允许 FULL
        if mode == AssistMode.FULL and not (has_project_snapshot or has_workspace):
            mode = AssistMode.LIGHT
            tool_set = LIGHT_TOOL_SET

        # 无工作区时 CHAT 也只走纯对话
        if mode == AssistMode.LIGHT and not has_workspace:
            mode = AssistMode.CHAT
            tool_set = frozenset()

        return mode, tool_set
