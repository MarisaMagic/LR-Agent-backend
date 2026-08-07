"""后端 MCP 客户端：连接前端 Electron MCP Server，动态获取工具并注入 Agent 工具集。

使用 langchain-mcp-adapters 将 MCP 工具 schema 转换为 LangChain StructuredTool。
MCP Server 地址由客户端通过 client_context.mcp_server_url 字段传入。

工具发现流程：
  1. 用 SSEConnection 连接 MCP Server
  2. 调用 load_mcp_tools() 获取所有工具
  3. 按 ToolCapability 去重后返回 list[StructuredTool]（不与内置工具能力冲突）

已知 MCP 工具（前端 server.ts 暴露）：
  - yolo_detect           本地 YOLO 推理
  - write_workspace_file  写工作区文本文件
  - list_project_images   枚举项目图片
  - memory_read           读取记忆 topic 文件
  - memory_write          写入记忆 topic 文件
  - read_agent_skill      读取全局 Agent Skill 的 SKILL.md 正文（走默认 SYNC runner，
                          与 canonical 工具无能力冲突，_infer_mcp_capability 返回 None）
"""

from __future__ import annotations

import logging

from langchain_core.tools import StructuredTool

from app.agent.tools.tool_registry_meta import (
    CANONICAL_CAPABILITIES,
    TOOL_CAPABILITY_MAP,
    ToolCapability,
)

logger = logging.getLogger(__name__)


def _infer_mcp_capability(tool_name: str) -> ToolCapability | None:
    """从 MCP 工具名称推断其能力类型。"""
    if tool_name in ("write_workspace_file",):
        return ToolCapability.WRITE_FILE
    if tool_name in ("yolo_detect",):
        return ToolCapability.AUTO_ANNOTATE
    if tool_name in ("list_project_images",):
        return ToolCapability.QUERY_CONTEXT
    return None


async def load_mcp_tools_from_server(
    mcp_server_url: str,
    *,
    existing_capabilities: set[ToolCapability] | None = None,
) -> list[StructuredTool]:
    """连接本地 MCP Server，动态发现并按能力去重后返回工具列表。

    失败时返回空列表（不影响 Agent 正常运行）。
    """
    if existing_capabilities is None:
        existing_capabilities = CANONICAL_CAPABILITIES

    try:
        from langchain_mcp_adapters.client import MultiServerMCPClient
    except ImportError:
        logger.warning("langchain-mcp-adapters 未安装，跳过 MCP 工具加载")
        return []

    sse_url = mcp_server_url.rstrip("/") + "/sse"
    try:
        client = MultiServerMCPClient(
            {
                "lr-agent-local": {
                    "url": sse_url,
                    "transport": "sse",
                }
            }
        )
        tools = await client.get_tools()

        # 按能力去重：跳过已有 canonical 实现的工具
        filtered: list[StructuredTool] = []
        skipped: list[str] = []
        for t in tools:
            cap = _infer_mcp_capability(t.name)
            if cap is not None and cap in existing_capabilities:
                skipped.append(t.name)
                continue
            filtered.append(t)

        logger.info(
            "MCP: 已从 %s 加载 %d 个工具：%s (跳过: %s)",
            mcp_server_url,
            len(filtered),
            [t.name for t in filtered],
            skipped or ["无"],
        )
        return filtered
    except Exception as exc:
        logger.warning("MCP: 连接 %s 失败，跳过 MCP 工具加载：%s", mcp_server_url, exc)
        return []
