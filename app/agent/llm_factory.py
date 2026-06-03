from langchain_openai import ChatOpenAI

from app.models.llm_provider import LlmProvider


def build_chat_model(
    provider: LlmProvider,
    api_key: str,
    *,
    streaming: bool = True,
    temperature: float = 0.7,
) -> ChatOpenAI:
    base_url = provider.base_url.rstrip("/")
    return ChatOpenAI(
        model=provider.model,
        api_key=api_key,
        base_url=base_url,
        streaming=streaming,
        temperature=temperature,
        timeout=120,
    )
