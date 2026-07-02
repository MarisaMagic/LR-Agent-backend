"""视觉预加载决策与执行：在 assist 开始时自动加载相关图片。"""

import uuid
from collections.abc import AsyncIterator, Callable

from langchain_core.messages import AIMessage, ToolMessage

from app.agent.assist_vision import (
    can_bootstrap_vision,
    pick_vision_relative_path,
    stream_vision_tool_execution,
    vision_relative_from_user_text,
)
from app.agent.chat_message_builder import build_multimodal_user_message
from app.agent.tools.workspace_file_reader import (
    VISION_TOOL_NAME,
    extract_vision_path_from_tool_result,
    format_vision_tool_result_for_display,
)
from app.core.config import Settings
from app.schemas.agent import ClientContextInput, StreamEventPayload


class VisionAutoLoader:
    """自动视觉预加载器。

    根据 user_content 判断是否需要预先加载图片，
    通过流式执行 vision tool 将图片注入消息上下文。
    """

    def __init__(
        self,
        vision_fn: Callable | None,
        provider_is_vision: bool,
        settings: Settings,
        client_context: ClientContextInput | None,
        user_content: str,
    ) -> None:
        self.vision_fn = vision_fn
        self.provider_is_vision = provider_is_vision
        self.settings = settings
        self.client_context = client_context
        self.user_content = user_content

    def should_load(self, is_resume: bool) -> bool:
        """判断是否需要视觉预加载。"""
        if is_resume or not self.provider_is_vision or self.vision_fn is None:
            return False
        return bool(vision_relative_from_user_text(self.user_content))

    async def try_bootstrap(
        self, messages: list
    ) -> AsyncIterator[StreamEventPayload]:
        """尝试在 assist 开始前预加载视觉图片。

        成功返回时将 vision_bootstrapped 设为 True 的标记通过
        传出 events 通知调用方。
        """
        rel = pick_vision_relative_path(self.client_context)
        if not rel and self.user_content.strip():
            rel = vision_relative_from_user_text(self.user_content)

        if can_bootstrap_vision(self.client_context, rel):
            tool_id = f"lr-vision-bootstrap-{uuid.uuid4().hex[:10]}"
            async for event in stream_vision_tool_execution(
                tool_id=tool_id,
                relative_path=rel,
                vision_fn=self.vision_fn,  # type: ignore[arg-type]
                provider_is_vision=self.provider_is_vision,
                settings=self.settings,
                messages=messages,
            ):
                yield event

    async def try_fallback(
        self, messages: list
    ) -> AsyncIterator[StreamEventPayload]:
        """在第一轮无 tool call 时尝试视觉回退（fallback）加载。"""
        rel = pick_vision_relative_path(self.client_context)
        if not rel and self.user_content.strip():
            rel = vision_relative_from_user_text(self.user_content)

        if can_bootstrap_vision(self.client_context, rel):
            tool_id = f"lr-vision-fallback-{uuid.uuid4().hex[:10]}"
            async for event in stream_vision_tool_execution(
                tool_id=tool_id,
                relative_path=rel,
                vision_fn=self.vision_fn,  # type: ignore[arg-type]
                provider_is_vision=self.provider_is_vision,
                settings=self.settings,
                messages=messages,
            ):
                yield event
