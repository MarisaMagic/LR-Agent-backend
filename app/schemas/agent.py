from typing import Any, Literal

from pydantic import BaseModel, Field


class ChatMessageInput(BaseModel):
    role: Literal["user", "assistant", "system"]
    content: str
    message_id: str | None = None
    interaction_mode: Literal["chat", "annotation"] | None = None


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


class AnnotationProjectSnapshotInput(BaseModel):
    project_id: str = Field(min_length=1, max_length=64)
    name: str = ""
    modality: str = ""
    annotation_type: str = ""
    labels: list[dict[str, Any]] = Field(default_factory=list)
    detection_models: list[dict[str, Any]] = Field(default_factory=list)
    project_directory_path: str | None = None


TurnKindLiteral = Literal[
    "execute_batch",
    "mutate_annotation",
    "analyze_data",
    "generate_report",
    "generate_document",
    "converse",
    "clarify_scope",
    "wants_batch",
    "unsupported",
]

TaskIntentLiteral = Literal[
    "converse",
    "query_annotation",
    "execute_batch",
    "mutate_annotation",
    "edit_annotation",
    "delete_annotation",
    "analyze_data",
    "generate_report",
    "generate_document",
    "clarify_scope",
    "wants_batch",
    "unsupported",
]


class TurnUnderstandingResultSchema(BaseModel):
    """Serialized turn understanding attached to client_context or API responses."""

    resolved_user_content: str = ""
    referenced_relative_paths: list[str] = Field(default_factory=list)
    resolved_active_relative_path: str | None = None
    task_intent: TaskIntentLiteral = "converse"
    turn_kind: TurnKindLiteral = "converse"
    needs_vision_input: bool = False
    confidence: float = Field(ge=0.0, le=1.0, default=0.8)
    scope_notes: str = ""
    reason: str = ""
    user_visible_hint: str | None = None


class ClientContextInput(BaseModel):
    workspace_root: str | None = None
    active_file_path: str | None = None
    active_relative_path: str | None = None
    project_directory_path: str | None = None
    active_annotation_project_id: str | None = None
    annotation_project_modality: str | None = None
    annotation_project_type: str | None = None
    agent_mode: Literal["chat", "annotation"] | None = None
    work_mode: Literal["editor", "annotation"] | None = None
    selected_annotation_id: str | None = None
    selected_annotation_ids: list[str] = Field(default_factory=list)
    annotation_project_snapshot: AnnotationProjectSnapshotInput | None = None
    turn_understanding: TurnUnderstandingResultSchema | None = None
    mcp_server_url: str | None = None


class TurnUnderstandRequest(BaseModel):
    provider_id: str = Field(min_length=1, max_length=64)
    user_content: str = Field(min_length=1, max_length=20_000)
    session_id: str | None = Field(default=None, max_length=64)
    user_message_id: str | None = Field(default=None, max_length=64)
    assistant_message_id: str | None = Field(default=None, max_length=64)
    truncate_from_message_id: str | None = Field(default=None, max_length=64)
    image_catalog_hint: list[str] | None = None
    client_context: ClientContextInput | None = None


class TurnUnderstandResponse(BaseModel):
    resolved_user_content: str
    referenced_relative_paths: list[str] = Field(default_factory=list)
    resolved_active_relative_path: str | None = None
    task_intent: TaskIntentLiteral = "converse"
    turn_kind: TurnKindLiteral = "converse"
    needs_vision_input: bool = False
    confidence: float = Field(ge=0.0, le=1.0, default=0.8)
    scope_notes: str = ""
    reason: str = ""
    user_visible_hint: str | None = None


class ClientToolResult(BaseModel):
    """客户端执行工具后返回的结果，随下一轮 /chat/stream 请求一并发送。"""

    tool_call_id: str = Field(min_length=1, max_length=128)
    name: str = Field(min_length=1, max_length=128)
    result: str = Field(description="工具执行结果（JSON 序列化字符串）")


class ChatStreamRequest(BaseModel):
    provider_id: str = Field(min_length=1, max_length=64)
    session_id: str = Field(min_length=1, max_length=64)
    client_job_id: str = Field(min_length=1, max_length=64)
    user_content: str = Field(min_length=1)
    user_message_id: str = Field(min_length=1, max_length=64)
    assistant_message_id: str = Field(min_length=1, max_length=64)
    truncate_from_message_id: str | None = None
    messages: list[ChatMessageInput] = Field(
        default_factory=list,
        description="登录态下由服务端从 DB 构建；客户端列表仅作兼容",
    )
    context: ChatContextInput | None = None
    client_context: ClientContextInput | None = None
    client_tool_results: list[ClientToolResult] = Field(
        default_factory=list,
        description="上一轮客户端工具执行结果，前端 resume 时携带",
    )


class AgentSessionCreateRequest(BaseModel):
    id: str | None = Field(default=None, max_length=64)
    title: str = Field(default="新对话", max_length=256)
    provider_id: str | None = Field(default=None, max_length=64)
    model: str | None = Field(default=None, max_length=128)
    annotation_project_id: str | None = Field(default=None, max_length=64)
    interaction_mode: Literal["chat", "annotation"] | None = None


class AgentMessageBlockPatchRequest(BaseModel):
    block_type: str | None = Field(default=None, max_length=64)
    block_index: int | None = Field(default=None, ge=0)
    patch: dict[str, Any] = Field(default_factory=dict)


class AnnotationRunStartRequest(BaseModel):
    provider_id: str = Field(min_length=1, max_length=64)
    session_id: str = Field(min_length=1, max_length=64)
    client_job_id: str = Field(min_length=1, max_length=64)
    user_content: str = Field(min_length=1)
    user_message_id: str = Field(min_length=1, max_length=64)
    assistant_message_id: str = Field(min_length=1, max_length=64)
    truncate_from_message_id: str | None = None
    client_context: ClientContextInput | None = None


class AnnotationRunEventsRequest(BaseModel):
    session_id: str = Field(min_length=1, max_length=64)
    assistant_message_id: str = Field(min_length=1, max_length=64)
    client_job_id: str = Field(min_length=1, max_length=64)
    events: list[dict[str, Any]] = Field(default_factory=list)
    seq: int | None = Field(default=None, ge=0)


class AnnotationRunFinalizeRequest(BaseModel):
    session_id: str = Field(min_length=1, max_length=64)
    assistant_message_id: str = Field(min_length=1, max_length=64)
    client_job_id: str = Field(min_length=1, max_length=64)
    status: Literal["done", "error", "stopped"] = "done"
    error: str | None = Field(default=None, max_length=4000)
    user_content: str | None = Field(default=None, max_length=20_000)


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
    interaction_mode: str | None = None
    provider_id: str = ""
    model: str = ""
    error: str | None = None
    created_at: int
    updated_at: int


class AgentSessionPublic(BaseModel):
    id: str
    title: str
    annotation_project_id: str | None = None
    interaction_mode: str | None = None
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


class ClientToolCallPayload(BaseModel):
    """client_tool_pending SSE 事件中单个客户端工具调用的描述。"""

    tool_call_id: str
    name: str
    arguments: dict[str, Any] = Field(default_factory=dict)


class StreamEventPayload(BaseModel):
    """SSE 流事件载荷。type 枚举：
      text_delta / reasoning_delta / tool_start / tool_result /
      preparing / context_updated / route_decided /
      file_proposal_start / file_proposal_delta / document_proposal / file_proposal /
      annotation_progress / annotation_proposal /
      analysis_script_proposal / tool_pending / client_tool_pending / error / done
    """

    type: str
    content: str | None = None
    stage: str | None = None
    status: str | None = None
    detail: str | None = None
    proposal: dict[str, Any] | None = None
    summary: str | None = None
    summary_up_to_message_id: str | None = None
    token_estimate: int | None = None
    tool_call_id: str | None = None
    name: str | None = None
    arguments: str | None = None
    result: str | None = None
    message: str | None = None
    image_path: str | None = None
    mode: str | None = None
    domain: str | None = None
    target: str | None = None
    reason: str | None = None
    # client_tool_pending 专用字段
    client_tool_calls: list[ClientToolCallPayload] | None = None

    @classmethod
    def from_client_dict(cls, data: dict[str, Any]) -> "StreamEventPayload":
        event_type = str(data.get("type") or "")
        content = data.get("content")
        detail = data.get("detail")
        message = data.get("message")
        if event_type == "analysis_script_proposal":
            content = data.get("script") or content
            detail = data.get("explanation") or detail
            message = data.get("error") or message
        image_path = data.get("imagePath") or data.get("image_path")
        summary = data.get("summary")
        if event_type in ("file_proposal_start", "document_proposal", "file_proposal"):
            content = data.get("content") or content
            detail = data.get("title") or detail
            image_path = data.get("suggestedRelativePath") or data.get("suggested_relative_path") or image_path
            summary = data.get("title") or summary
        domain = data.get("domain")
        if event_type == "annotation_progress":
            domain = data.get("pipelineKind") or data.get("pipeline_kind") or domain
        return cls(
            type=event_type,
            content=content,
            stage=data.get("stage"),
            status=data.get("status"),
            detail=detail,
            proposal=data.get("proposal"),
            summary=summary,
            summary_up_to_message_id=data.get("summaryUpToMessageId") or data.get("summary_up_to_message_id"),
            token_estimate=data.get("tokenEstimate") or data.get("token_estimate"),
            tool_call_id=data.get("toolCallId") or data.get("tool_call_id"),
            name=data.get("name"),
            arguments=data.get("arguments"),
            result=data.get("result"),
            message=message,
            image_path=image_path,
            mode=data.get("mode"),
            domain=domain,
            target=data.get("target"),
            reason=data.get("reason"),
        )

    def to_sse_dict(self) -> dict[str, Any]:
        data: dict[str, Any] = {"type": self.type}
        if self.content is not None:
            data["content"] = self.content
        if self.stage is not None:
            data["stage"] = self.stage
        if self.status is not None:
            data["status"] = self.status
        if self.detail is not None:
            data["detail"] = self.detail
        if self.proposal is not None:
            data["proposal"] = self.proposal
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
        if self.image_path is not None:
            if self.type in ("file_proposal_start", "file_proposal_delta", "file_proposal", "document_proposal"):
                data["suggestedRelativePath"] = self.image_path
            else:
                data["imagePath"] = self.image_path
        if self.type in ("file_proposal_start", "file_proposal", "document_proposal"):
            if self.summary is not None:
                data["title"] = self.summary
            if self.detail is not None and "title" not in data:
                data["title"] = self.detail
        if self.type == "analysis_script_proposal":
            if self.content is not None:
                data["script"] = self.content
            if self.detail is not None:
                data["explanation"] = self.detail
            if self.message is not None:
                data["error"] = self.message
        if self.mode is not None:
            data["mode"] = self.mode
        if self.domain is not None:
            data["domain"] = self.domain
        if self.target is not None:
            data["target"] = self.target
        if self.reason is not None:
            data["reason"] = self.reason
        if self.client_tool_calls is not None:
            serialized = [
                {
                    "toolCallId": c.tool_call_id,
                    "name": c.name,
                    "arguments": c.arguments,
                }
                for c in self.client_tool_calls
            ]
            data["clientToolCalls"] = serialized
            data["toolCalls"] = serialized
        return data
