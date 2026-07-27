from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, Field


class JobState(str, Enum):
    """Agent 任务生命周期状态。"""

    REGISTERED = "registered"
    STREAMING = "streaming"
    TOOL_PENDING = "tool_pending"
    RESUMING = "resuming"
    DONE = "done"
    ERROR = "error"
    CANCELLED = "cancelled"


class ChatMessageInput(BaseModel):
    role: Literal["user", "assistant", "system"]
    content: str
    message_id: str | None = None
    interaction_mode: Literal["chat", "annotation"] | None = None


class AnnotationProjectSnapshotInput(BaseModel):
    project_id: str = Field(min_length=1, max_length=64)
    name: str = ""
    modality: str = ""
    annotation_type: str = ""
    labels: list[dict[str, Any]] = Field(default_factory=list)
    detection_models: list[dict[str, Any]] = Field(default_factory=list)
    project_directory_path: str | None = None


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
    mcp_server_url: str | None = None


class ClientToolResult(BaseModel):
    """客户端执行工具后返回的结果，随下一轮 /chat/stream 请求一并发送。"""

    tool_call_id: str = Field(min_length=1, max_length=128)
    name: str = Field(min_length=1, max_length=128)
    result: str = Field(description="工具执行结果（JSON 序列化字符串）")


class LocalChatStreamRequest(BaseModel):
    """Lightweight chat/stream request — no DB dependency."""

    api_key: str = Field(min_length=1)
    base_url: str = Field(min_length=1)
    model: str = Field(min_length=1)
    supports_vision: bool = False
    messages: list[ChatMessageInput]
    user_content: str = Field(min_length=1)
    system_prompt: str | None = None
    context_summary: str | None = None
    context_summary_up_to_message_id: str | None = None
    client_context: ClientContextInput | None = None
    client_tool_results: list[ClientToolResult] = Field(default_factory=list)
    client_job_id: str = Field(min_length=1, max_length=64)


class ChatCancelRequest(BaseModel):
    client_job_id: str = Field(min_length=1, max_length=64)


class ClientToolCallPayload(BaseModel):
    """tool_pending SSE 事件中单个客户端工具调用的描述。"""

    tool_call_id: str
    name: str
    arguments: dict[str, Any] = Field(default_factory=dict)


class StreamEventPayload(BaseModel):
    """SSE 流事件载荷。"""

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
    client_tool_calls: list[ClientToolCallPayload] | None = None

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
                    "toolCallId": call.tool_call_id,
                    "name": call.name,
                    "arguments": call.arguments,
                }
                for call in self.client_tool_calls
            ]
            data["clientToolCalls"] = serialized
            data["toolCalls"] = serialized
        return data
