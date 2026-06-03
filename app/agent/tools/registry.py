from collections.abc import Callable
from typing import Any

from langchain_core.tools import StructuredTool

from app.agent.tools.help import get_lr_agent_help
from app.models.user import User
from app.schemas.agent import ClientContextInput


def build_tools_p1(
    user: User,
    client_context: ClientContextInput | None,
) -> list[StructuredTool]:
    def account_summary() -> str:
        verified = "已验证" if user.email_verified else "未验证"
        name = user.display_name or user.username or "未设置"
        return (
            f"邮箱: {user.email} ({verified})\n"
            f"显示名: {name}\n"
            f"用户名: {user.username or '未设置'}"
        )

    def describe_context() -> str:
        if client_context is None:
            return "客户端未提供当前界面上下文。"
        parts: list[str] = []
        if client_context.workspace_root:
            parts.append(f"工作区根目录: {client_context.workspace_root}")
        if client_context.active_file_path:
            parts.append(f"当前打开文件: {client_context.active_file_path}")
        if client_context.active_annotation_project_id:
            parts.append(f"当前标注项目 ID: {client_context.active_annotation_project_id}")
        if not parts:
            return "工作区已连接，但未打开具体文件或标注项目。"
        return "\n".join(parts)

    def help_tool(topic: str | None = None) -> str:
        return get_lr_agent_help(topic)

    return [
        StructuredTool.from_function(
            func=account_summary,
            name="get_account_summary",
            description="获取当前登录用户的账户摘要（邮箱、验证状态、显示名）",
        ),
        StructuredTool.from_function(
            func=help_tool,
            name="get_lr_agent_help",
            description="获取 LR-Agent 应用功能说明，可选 topic 关键词",
        ),
        StructuredTool.from_function(
            func=describe_context,
            name="describe_client_context",
            description="描述用户当前 Electron 客户端界面上下文（工作区、打开文件、标注项目）",
        ),
    ]


def tool_fn_map(tools: list[StructuredTool]) -> dict[str, Callable[..., Any]]:
    return {tool.name: tool.func for tool in tools if tool.func is not None}
