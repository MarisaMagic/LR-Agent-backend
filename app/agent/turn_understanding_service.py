"""Unified per-turn understanding: LLM-only deixis, intent, and routing."""

from __future__ import annotations

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

from app.agent.annotation.llm_invoke import invoke_json_model
from app.agent.context_helpers import (
    client_active_relative,
    label_names_from_context,
    normalize_rel_path,
    project_context_lines,
    project_directory,
)
from app.schemas.agent import ClientContextInput, TaskIntentLiteral, TurnKindLiteral

UNDERSTAND_SYSTEM = """你是 LR-Agent 的回合理解与路由助手。根据对话历史、当前用户消息与界面状态，**一次**输出结构化 JSON。

职责（单次完成，勿分步）：
1. 指代消解：结合历史与 active_relative_path，解析「这两张/上面那些/漏了另一张/按前述」等为具体 relative_path
2. 路由 turn_kind（由你全权决定，不受 UI 模式开关约束）：
   - execute_batch：检测、标注、批量处理、补标、纠正遗漏图片等需要启动标注流水线的动作
   - converse：问答、解释策略、查已有标注（可读 JSON）、寒暄、需看图描述的内容
   - clarify_scope：与标注相关但图片范围仍不明确，需先追问用户
   - unsupported：当前项目类型无法执行（极少）
3. needs_vision_input：必须看像素才能回答（认人、数人等），且不是读取已有标注 JSON 可解决的；为 true 时对话 Agent 应调用 read_image_for_vision

规则：
- referenced_relative_paths：本轮要处理的**全部**图片相对路径，可多个
- 用户说「这两张」且历史提到过 data/1.jpg 与 data/2.jpg → 两个路径都要列入
- 用户反馈「漏了/还有一张/另一张呢」→ 结合历史补全缺失路径，通常 turn_kind=execute_batch
- resolved_user_content：补全指代后的完整中文句，可独立理解
- resolved_active_relative_path：UI/指代绑定的主图，无则 null
- 不得虚构候选中不存在的路径
- 回复中禁止出现 [Ask]、[Agent] 等模式前缀格式

输出 JSON 字段：
- resolved_user_content
- referenced_relative_paths: string[]
- resolved_active_relative_path: string | null
- task_intent: converse|query_annotation|execute_batch|clarify_scope|unsupported
- turn_kind: execute_batch|converse|clarify_scope|unsupported
- needs_vision_input: bool
- confidence: 0~1
- scope_notes: 简短中文
- reason: 简短中文
- user_visible_hint: string | null（可选）"""


class TurnUnderstandingLlmResult(BaseModel):
    resolved_user_content: str = Field(min_length=1, max_length=20_000)
    referenced_relative_paths: list[str] = Field(default_factory=list)
    resolved_active_relative_path: str | None = None
    task_intent: TaskIntentLiteral = "converse"
    turn_kind: TurnKindLiteral = "converse"
    needs_vision_input: bool = False
    confidence: float = Field(ge=0.0, le=1.0, default=0.8)
    scope_notes: str = ""
    reason: str = ""
    user_visible_hint: str | None = None


class TurnUnderstandingResult(BaseModel):
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


_VALID_TURN_KINDS: frozenset[str] = frozenset(
    {"execute_batch", "converse", "clarify_scope", "unsupported", "wants_batch"},
)


def _normalize_turn_kind(raw: str) -> TurnKindLiteral:
    if raw == "wants_batch":
        return "execute_batch"
    if raw in _VALID_TURN_KINDS:
        return raw  # type: ignore[return-value]
    return "converse"


def _build_understand_human(
    *,
    conversation_transcript: str,
    current_user_message: str,
    client_context: ClientContextInput | None,
    provider_is_vision: bool,
    image_catalog_hint: list[str] | None,
) -> str:
    transcript_block = conversation_transcript.strip() or "（无历史对话）"
    active_rel = client_active_relative(client_context) or "（无）"
    project_dir = project_directory(client_context) or "（无）"
    project_lines = project_context_lines(client_context)
    labels = ", ".join(label_names_from_context(client_context)) or "（无）"
    catalog = ""
    if image_catalog_hint:
        sample = image_catalog_hint[:50]
        catalog = f"\n候选图片路径（摘要，最多50）：{', '.join(sample)}\n"

    ui_hint = ""
    if client_context and client_context.agent_mode:
        ui_hint = f"\n客户端 UI 模式（仅供参考，不得以此代替路由决策）：{client_context.agent_mode}\n"

    return (
        f"provider_is_vision: {provider_is_vision}\n"
        f"project_directory_path: {project_dir}\n"
        f"active_relative_path: {active_rel}\n"
        f"项目标签: {labels}\n"
        f"{project_lines}\n"
        f"{catalog}"
        f"{ui_hint}\n"
        f"【对话历史】\n{transcript_block}\n\n"
        f"【当前用户消息】\n{current_user_message.strip()}"
    )


def _llm_to_result(parsed: TurnUnderstandingLlmResult) -> TurnUnderstandingResult:
    paths = [normalize_rel_path(p) for p in parsed.referenced_relative_paths if p.strip()]
    paths = list(dict.fromkeys(paths))
    active = parsed.resolved_active_relative_path
    if active:
        active = normalize_rel_path(active)
    turn_kind = _normalize_turn_kind(str(parsed.turn_kind))
    return TurnUnderstandingResult(
        resolved_user_content=parsed.resolved_user_content.strip(),
        referenced_relative_paths=paths,
        resolved_active_relative_path=active,
        task_intent=parsed.task_intent or turn_kind,
        turn_kind=turn_kind,
        needs_vision_input=parsed.needs_vision_input,
        confidence=parsed.confidence,
        scope_notes=parsed.scope_notes,
        reason=parsed.reason or "llm",
        user_visible_hint=parsed.user_visible_hint,
    )


async def understand_turn(
    llm: ChatOpenAI,
    *,
    user_content: str,
    client_context: ClientContextInput | None = None,
    conversation_transcript: str = "",
    provider_is_vision: bool = False,
    image_catalog_hint: list[str] | None = None,
) -> TurnUnderstandingResult:
    """Single LLM call: deixis resolution + intent + route."""
    human = _build_understand_human(
        conversation_transcript=conversation_transcript,
        current_user_message=user_content,
        client_context=client_context,
        provider_is_vision=provider_is_vision,
        image_catalog_hint=image_catalog_hint,
    )
    parsed = await invoke_json_model(
        llm,
        [SystemMessage(content=UNDERSTAND_SYSTEM), HumanMessage(content=human)],
        TurnUnderstandingLlmResult,
    )
    return _llm_to_result(parsed)


def format_understanding_for_system_prompt(result: TurnUnderstandingResult) -> str:
    paths = ", ".join(result.referenced_relative_paths) or "（无）"
    active = result.resolved_active_relative_path or "（无）"
    return (
        "【回合理解】\n"
        f"- 解析后用户意图：{result.resolved_user_content}\n"
        f"- 涉及图片路径：{paths}\n"
        f"- 当前指代图片：{active}\n"
        f"- 路由：{result.turn_kind}\n"
        f"- 需要看图描述：{'是，对话 Agent 应调用 read_image_for_vision（调用后系统会注入附图）' if result.needs_vision_input else '否'}\n"
        f"- 说明：{result.scope_notes or result.reason}"
    )


# Re-export for orchestrator vision path resolution
_project_directory = project_directory
