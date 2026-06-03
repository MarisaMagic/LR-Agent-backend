from typing import Any, Literal

from pydantic import BaseModel, Field


class ChatMessageInput(BaseModel):
    role: Literal["user", "assistant", "system"]
    content: str


class ChatContextConfigInput(BaseModel):
    max_context_tokens: int = 12_000
    reserve_completion_tokens: int = 2_048
    max_turns_in_window: int = 20
    summarize_trigger_ratio: float = 0.85
    min_turns_before_summarize: int = 6


class ChatContextInput(BaseModel):
    summary: str | None = None
    summary_up_to_message_id: str | None = None
    config: ChatContextConfigInput | None = None


class ClientContextInput(BaseModel):
    workspace_root: str | None = None
    active_file_path: str | None = None
    active_annotation_project_id: str | None = None


class ChatStreamRequest(BaseModel):
    provider_id: str = Field(min_length=1, max_length=64)
    session_id: str = Field(min_length=1, max_length=64)
    client_job_id: str = Field(min_length=1, max_length=64)
    user_content: str = Field(min_length=1)
    user_message_id: str = Field(min_length=1, max_length=64)
    assistant_message_id: str = Field(min_length=1, max_length=64)
    truncate_from_message_id: str | None = None
    messages: list[ChatMessageInput] = Field(default_factory=list)
    context: ChatContextInput | None = None
    client_context: ClientContextInput | None = None


class AgentSessionCreateRequest(BaseModel):
    id: str | None = Field(default=None, max_length=64)
    title: str = Field(default="新对话", max_length=256)
    provider_id: str | None = Field(default=None, max_length=64)
    model: str | None = Field(default=None, max_length=128)


class AgentSessionPatchRequest(BaseModel):
    title: str | None = Field(default=None, max_length=256)
    provider_id: str | None = Field(default=None, max_length=64)
    model: str | None = Field(default=None, max_length=128)


class AgentMessagePublic(BaseModel):
    id: str
    session_id: str
    role: str
    blocks: list[dict[str, Any]]
    status: str
    provider_id: str = ""
    model: str = ""
    error: str | None = None
    created_at: int
    updated_at: int


class AgentSessionPublic(BaseModel):
    id: str
    title: str
    provider_id: str = ""
    model: str = ""
    message_ids: list[str] = Field(default_factory=list)
    message_count: int = 0
    last_message_preview: str | None = None
    context_summary: str | None = None
    summary_up_to_message_id: str | None = None
    last_context_token_estimate: int | None = None
    created_at: int
    updated_at: int


class AgentSessionListResponse(BaseModel):
    sessions: list[AgentSessionPublic]
    next_cursor: str | None = None
    has_more: bool = False


class AgentSessionDetailResponse(BaseModel):
    session: AgentSessionPublic
    messages: list[AgentMessagePublic]
    has_more_before: bool = False


class ChatCancelRequest(BaseModel):
    client_job_id: str = Field(min_length=1, max_length=64)


class StreamEventPayload(BaseModel):
    type: str
    content: str | None = None
    stage: str | None = None
    summary: str | None = None
    summary_up_to_message_id: str | None = None
    token_estimate: int | None = None
    tool_call_id: str | None = None
    name: str | None = None
    arguments: str | None = None
    result: str | None = None
    message: str | None = None
    mode: str | None = None
    domain: str | None = None

    def to_sse_dict(self) -> dict[str, Any]:
        data: dict[str, Any] = {"type": self.type}
        if self.content is not None:
            data["content"] = self.content
        if self.stage is not None:
            data["stage"] = self.stage
        if self.summary is not None:
            data["summary"] = self.summary
        if self.summary_up_to_message_id is not None:
            data["summaryUpToMessageId"] = self.summary_up_to_message_id
        if self.token_estimate is not None:
            data["tokenEstimate"] = self.token_estimate
        if self.tool_call_id is not None:
            data["toolCallId"] = self.tool_call_id
        if self.name is not None:
            data["name"] = self.name
        if self.arguments is not None:
            data["arguments"] = self.arguments
        if self.result is not None:
            data["result"] = self.result
        if self.message is not None:
            data["message"] = self.message
        if self.mode is not None:
            data["mode"] = self.mode
        if self.domain is not None:
            data["domain"] = self.domain
        return data
