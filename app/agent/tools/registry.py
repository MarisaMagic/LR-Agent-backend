"""Assist 模式工具注册表。

由 orchestrator 调用 build_tools_p1 构建工具列表，assist_service 通过 tool_fn_map 按名称执行。
工具分三类：
  - 上下文查询：账户、帮助、界面状态、标注项目快照
  - 文件读取/写入：文本/代码、文档、图片（视觉）、已有标注 JSON、写文件提案
  - 异步工具（ASYNC_TOOL_NAMES）：由前端 Electron 执行；调度器发出 tool_pending SSE 并暂停 loop
"""

import json
from collections.abc import Callable
from typing import Any

from langchain_core.tools import StructuredTool

# 客户端工具名称集合（向后兼容）：由 tool_registry_meta 统一定义
from app.agent.tools.tool_registry_meta import ASYNC_TOOL_NAMES, CLIENT_TOOL_NAMES

from app.agent.annotation.annotation_doc_reader import read_file_annotation_doc
from app.agent.context_helpers import project_directory
from app.agent.context_snapshot import format_snapshot_for_prompt
from app.agent.tools.help import get_lr_agent_help
from app.agent.tools.workspace_file_reader import (
    read_document_file,
    read_image_for_vision_tool,
    read_workspace_text_file,
    write_workspace_file_tool,
)
from app.core.config import Settings
from app.models.user import User
from app.schemas.agent import ClientContextInput

ANNOTATION_TOOL_NAMES: frozenset[str] = frozenset(
    {
        "describe_annotation_project",
        "read_file_annotation",
        "execute_batch_annotation",
        "mutate_annotation",
        "analyze_data",
    }
)


def _client_tool_stub(tool_name: str) -> StructuredTool:
    """返回一个客户端工具的 schema 存根（func 不会被本地调用）。"""
    # assist_service 在执行工具前会先检测 CLIENT_TOOL_NAMES，拦截并发出 client_tool_pending
    def _unreachable(**_kwargs: object) -> str:  # noqa: ANN001
        return f"[{tool_name}] 此工具应由前端执行，本地调用不应发生。"
    _unreachable.__name__ = tool_name
    return _unreachable


def build_tools_p1(
    user: User,
    client_context: ClientContextInput | None,
    *,
    settings: Settings,
    provider_is_vision: bool = False,
) -> list[StructuredTool]:
    """构建工具集：只读工具 + 写文件提案工具 + 客户端工具 schema 存根。"""
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
        if client_context.work_mode:
            parts.append(f"工作模式: {client_context.work_mode}")
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
        project_dir = project_directory(client_context)
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

    def write_file(relative_path: str, content: str) -> str:
        return write_workspace_file_tool(client_context, relative_path, content)

    # Phase 1 工具集（只读 + 写文件提案）
    tools = [
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
        StructuredTool.from_function(
            func=write_file,
            name="write_workspace_file",
            description=(
                "在工作区内创建或覆写文本/代码文件。"
                "支持 .md .txt .json .yaml .py .cpp .ts 等格式；父目录不存在时会自动创建。"
                "调用后生成 file_proposal，用户点击 Keep All 后才实际写盘。"
                "relative_path 示例：reports/summary.md、src/dijkstra.cpp。"
                "content 为完整文件内容。"
                "未成功调用本工具前，禁止在回复中声称文件已写入。"
            ),
        ),
        # ── 客户端工具（schema 存根，实现体在前端 Electron 进程）────────────
        StructuredTool.from_function(
            func=_client_tool_stub("execute_batch_annotation"),
            name="execute_batch_annotation",
            description=(
                "【客户端工具】对标注项目中的图片批量运行目标检测并自动标注。"
                "前端将启动 YOLO 推理 → 标签映射 → 生成标注提案等完整流水线。"
                "调用前不要 read_image_for_vision。"
                "user_request：必须原样传递用户原话（如「标注 data 文件夹下所有图片」），"
                "不要改写为单张路径，不要自行指定标签 ID。"
                "scope_hint（可选）：范围补充说明（如「仅限子目录 train/」）。"
                "必须发起真实 tool call，正文伪代码无效。"
            ),
        ),
        StructuredTool.from_function(
            func=_client_tool_stub("mutate_annotation"),
            name="mutate_annotation",
            description=(
                "【客户端工具】修改或删除项目中已有的标注框（改标签、删框、批量纠错）。"
                "不包含新增检测框；如需新增请使用 execute_batch_annotation。"
                "user_request：用户原始请求（如「把所有 dog 标签改为 puppy」）。"
                "调用后前端生成变更提案，用户确认后执行写入。"
            ),
        ),
        StructuredTool.from_function(
            func=_client_tool_stub("analyze_data"),
            name="analyze_data",
            description=(
                "【客户端工具】对当前标注项目的标注数据进行统计或分布分析。"
                "前端将生成 Python 分析脚本并在沙箱中执行，返回统计结果（图表/表格）。"
                "user_request：分析需求（如「各类别标注数量分布」「IoU 分布直方图」）。"
                "建议先调用此工具获取数据，再结合 write_workspace_file 写入报告或导出文件。"
            ),
        ),
    ]

    if client_context and client_context.work_mode == "editor":
        tools = [tool for tool in tools if tool.name not in ANNOTATION_TOOL_NAMES]

    return tools


def tool_fn_map(tools: list[StructuredTool]) -> dict[str, Callable[..., Any]]:
    """将 StructuredTool 列表转为 name → func 映射，供 assist_service 本地执行。"""
    return {tool.name: tool.func for tool in tools if tool.func is not None}
