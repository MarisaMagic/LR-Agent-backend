import json
from collections.abc import Callable
from typing import Any

from langchain_core.tools import StructuredTool

from app.agent.annotation.annotation_doc_reader import read_file_annotation_doc
from app.agent.context_snapshot import format_snapshot_for_prompt
from app.agent.tools.help import get_lr_agent_help
from app.agent.tools.workspace_file_reader import (
    read_document_file,
    read_image_for_vision_tool,
    read_workspace_text_file,
)
from app.agent.turn_understanding_service import _project_directory
from app.core.config import Settings
from app.models.user import User
from app.schemas.agent import ClientContextInput


def build_tools_p1(
    user: User,
    client_context: ClientContextInput | None,
    *,
    settings: Settings,
    provider_is_vision: bool = False,
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
        if client_context.active_relative_path:
            parts.append(f"当前打开文件（相对路径）: {client_context.active_relative_path}")
        if client_context.active_annotation_project_id:
            parts.append(f"当前标注项目 ID: {client_context.active_annotation_project_id}")
        if client_context.annotation_project_modality:
            parts.append(f"任务模态: {client_context.annotation_project_modality}")
        if client_context.annotation_project_type:
            parts.append(f"标注类型: {client_context.annotation_project_type}")
        if client_context.agent_mode:
            parts.append(f"交互模式: {client_context.agent_mode}")
        if not parts:
            return "工作区已连接，但未打开具体文件或标注项目。"
        return "\n".join(parts)

    def describe_annotation_project() -> str:
        if client_context is None or client_context.annotation_project_snapshot is None:
            return "当前未绑定标注项目快照。请确认用户已在标注任务中打开项目。"
        return format_snapshot_for_prompt(client_context.annotation_project_snapshot)

    def help_tool(topic: str | None = None) -> str:
        return get_lr_agent_help(topic)

    def read_file_annotation(relative_path: str) -> str:
        project_dir = _project_directory(client_context)
        if not project_dir:
            return "无法读取标注：未绑定项目目录。请确认已在标注任务中打开项目。"
        doc, err = read_file_annotation_doc(project_dir, relative_path)
        if doc is None:
            return err or "未找到标注。"
        return json.dumps(doc, ensure_ascii=False, indent=2)[:12_000]

    def read_workspace_file(relative_path: str = "") -> str:
        return read_workspace_text_file(
            client_context,
            relative_path,
            settings=settings,
        )

    def read_image_for_vision(relative_path: str = "") -> str:
        return read_image_for_vision_tool(
            client_context,
            relative_path,
            provider_is_vision=provider_is_vision,
        )

    def read_document(relative_path: str = "") -> str:
        return read_document_file(client_context, relative_path, settings=settings)

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
        StructuredTool.from_function(
            func=describe_annotation_project,
            name="describe_annotation_project",
            description="获取当前标注项目的标签树、任务类型、模态与可用检测模型列表",
        ),
        StructuredTool.from_function(
            func=read_file_annotation,
            name="read_file_annotation",
            description=(
                "读取项目中某张图片的已有标注 JSON（相对路径，如 data/2.jpg）。"
                "用于回答「标了谁」「有哪些框」等查询。"
            ),
        ),
        StructuredTool.from_function(
            func=read_workspace_file,
            name="read_workspace_file",
            description=(
                "读取工作区或项目内的文本/代码文件内容（如 .py .ts .md .json .yaml .txt）。"
                "relative_path 为空时使用当前打开文件。"
            ),
        ),
        StructuredTool.from_function(
            func=read_image_for_vision,
            name="read_image_for_vision",
            description=(
                "加载图片并在调用成功后由系统注入附图，供你直接根据像素回答"
                "（场景、物体、人数、文字 OCR、外观等）。relative_path 为空时使用当前打开的图片。"
                "需视觉探针通过；查已有标注 JSON 请用 read_file_annotation，勿用本工具代替。"
            ),
        ),
        StructuredTool.from_function(
            func=read_document,
            name="read_document_file",
            description=(
                "提取 PDF 或 DOCX 文档正文。"
                "relative_path 为空时使用当前打开的文件。"
            ),
        ),
    ]


def tool_fn_map(tools: list[StructuredTool]) -> dict[str, Callable[..., Any]]:
    return {tool.name: tool.func for tool in tools if tool.func is not None}
