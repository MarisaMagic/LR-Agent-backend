from collections.abc import AsyncIterator

from langchain_openai import ChatOpenAI

from app.agent.stream_adapter import events_from_chunk
from app.schemas.agent import StreamEventPayload


async def stream_chat(
    llm: ChatOpenAI,
    lc_messages: list,
) -> AsyncIterator[StreamEventPayload]:
    yield StreamEventPayload(type="preparing", stage="streaming")
    async for chunk in llm.astream(lc_messages):
        for event in events_from_chunk(chunk, emit_tool_chunks=False):
            yield event
