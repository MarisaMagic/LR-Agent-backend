"""LLM 适配器抽象接口——为未来换 SDK 预留隔离边界。

当前实现委托给 LangChain，但所有 core 模块通过本接口导入。
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import AsyncIterator
from dataclasses import dataclass
from typing import Any


@dataclass
class StreamChunk:
    """流式输出的单个 chunk。"""
    content: str = ""
    reasoning: str = ""
    tool_call_chunks: list[dict[str, Any]] | None = None
    tool_calls: list[dict[str, Any]] | None = None


class ChatModelAdapter(ABC):
    """对话模型适配器抽象。"""

    @abstractmethod
    async def astream(self, messages: list[Any]) -> AsyncIterator[StreamChunk]:
        """流式推理。"""
        ...

    @abstractmethod
    def bind_tools(self, tools: list[ToolAdapter]) -> "ChatModelAdapter":
        """绑定工具并返回新的适配器实例。"""
        ...

    @abstractmethod
    async def ainvoke(self, messages: list[Any]) -> StreamChunk:
        """单次推理（非流式）。"""
        ...


class ToolAdapter(ABC):
    """工具适配器抽象。"""

    @property
    @abstractmethod
    def name(self) -> str:
        ...

    @property
    @abstractmethod
    def description(self) -> str:
        ...

    @abstractmethod
    def invoke(self, **kwargs: Any) -> str:
        """同步执行工具。"""
        ...
