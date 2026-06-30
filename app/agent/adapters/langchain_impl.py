"""LangChain 适配器实现——将 ChatOpenAI / StructuredTool 封装为 ChatModelAdapter / ToolAdapter。

当前为薄封装，直接委托给 LangChain 实例。
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any

from langchain_core.messages import AIMessageChunk
from langchain_core.tools import StructuredTool
from langchain_openai import ChatOpenAI

from app.agent.adapters.interfaces import ChatModelAdapter, StreamChunk, ToolAdapter


class LangChainChatAdapter(ChatModelAdapter):
    """LangChain ChatOpenAI 适配器。"""

    def __init__(self, llm: ChatOpenAI) -> None:
        self._llm = llm

    async def astream(self, messages: list[Any]) -> AsyncIterator[StreamChunk]:
        async for chunk in self._llm.astream(messages):
            yield self._to_stream_chunk(chunk)

    def bind_tools(self, tools: list[ToolAdapter]) -> ChatModelAdapter:
        lc_tools = [
            t._structured_tool for t in tools
            if isinstance(t, LangChainToolAdapter)
        ]
        return LangChainChatAdapter(self._llm.bind_tools(lc_tools))

    async def ainvoke(self, messages: list[Any]) -> StreamChunk:
        result = await self._llm.ainvoke(messages)
        return self._to_stream_chunk(result)

    @staticmethod
    def _to_stream_chunk(chunk: AIMessageChunk) -> StreamChunk:
        content = chunk.content
        if isinstance(content, str):
            text = content
        elif isinstance(content, list):
            text = "".join(
                p.get("text", "") for p in content if isinstance(p, dict)
            )
        else:
            text = str(content) if content else ""

        raw_reasoning = (chunk.additional_kwargs or {}).get("reasoning_content")

        return StreamChunk(
            content=text,
            reasoning=str(raw_reasoning) if raw_reasoning else "",
            tool_call_chunks=list(chunk.tool_call_chunks or []),
            tool_calls=list(getattr(chunk, "tool_calls", None) or []),
        )


class LangChainToolAdapter(ToolAdapter):
    """LangChain StructuredTool 适配器。"""

    def __init__(self, tool: StructuredTool) -> None:
        self._structured_tool = tool

    @property
    def name(self) -> str:
        return self._structured_tool.name

    @property
    def description(self) -> str:
        return self._structured_tool.description

    def invoke(self, **kwargs: Any) -> str:
        result = self._structured_tool.invoke(kwargs)
        return str(result) if result is not None else ""


def wrap_chat_model(llm: ChatOpenAI) -> ChatModelAdapter:
    """将 ChatOpenAI 封装为 ChatModelAdapter。"""
    return LangChainChatAdapter(llm)


def wrap_tool(tool: StructuredTool) -> ToolAdapter:
    """将 StructuredTool 封装为 ToolAdapter。"""
    return LangChainToolAdapter(tool)
