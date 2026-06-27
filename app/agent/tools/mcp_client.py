"""后端 MCP 客户端：连接前端 Electron MCP Server，动态获取工具并注入 Agent 工具集。

使用 langchain-mcp-adapters 将 MCP 工具 schema 转换为 LangChain StructuredTool。
MCP Server 地址由客户端通过 client_context.mcp_server_url 字段传入。

工具发现流程：
  1. 用 SSEConnection 连接 MCP Server
  2. 调用 load_mcp_tools() 获取所有工具
  3. 返回 list[StructuredTool]，由 orchestrator 注入 build_tools_p1 列表

已知 MCP 工具（前端 server.ts 暴露）：
  - yolo_detect           本地 YOLO 推理
  - write_workspace_file  写工作区文本文件
  - list_project_images   枚举项目图片
"""

from __future__ import annotations

import logging

from langchain_core.tools import StructuredTool

logger = logging.getLogger(__name__)


async def load_mcp_tools_from_server(mcp_server_url: str) -> list[StructuredTool]:
    """连接本地 MCP Server，动态发现并返回工具列表。

    失败时返回空列表（不影响 Agent 正常运行）。
    """
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
        logger.info(
            "MCP: 已从 %s 加载 %d 个工具：%s",
            mcp_server_url,
            len(tools),
            [t.name for t in tools],
        )
        return list(tools)
    except Exception as exc:
        logger.warning("MCP: 连接 %s 失败，跳过 MCP 工具加载：%s", mcp_server_url, exc)
        return []
