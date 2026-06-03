from collections.abc import AsyncIterator

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

from app.agent.context_service import SUMMARIZE_PROMPT, messages_for_summary
from app.agent.stream_adapter import events_from_chunk
from app.schemas.agent import ChatMessageInput, StreamEventPayload


async def stream_chat(
    llm: ChatOpenAI,
    lc_messages: list,
) -> AsyncIterator[StreamEventPayload]:
    yield StreamEventPayload(type="preparing", stage="streaming")
    async for chunk in llm.astream(lc_messages):
        for event in events_from_chunk(chunk):
            yield event


async def summarize_messages(
    llm: ChatOpenAI,
    messages: list[ChatMessageInput],
) -> str:
    history = messages_for_summary(messages)
    result = await llm.ainvoke(
        [
            SystemMessage(content=SUMMARIZE_PROMPT),
            HumanMessage(content=history),
        ],
    )
    content = result.content
    if isinstance(content, str):
        return content.strip()
    return str(content)
