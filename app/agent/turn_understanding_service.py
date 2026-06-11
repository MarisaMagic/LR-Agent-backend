"""回合理解服务：单次 LLM 调用完成指代消解、意图识别与路由决策。

由 orchestrator 在 assist 模式下调用（客户端未预计算时），也可通过 API 独立调用。
理解结果流向：
  1. format_understanding_for_system_prompt → 注入对话 Agent 系统提示词
  2. assist_vision.pick_vision_relative_path → 视觉预加载路径选择
  3. 客户端 turn_understanding 字段 → orchestrator 直接复用，跳过 LLM 调用
"""

from __future__ import annotations

import logging

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field, field_validator

from app.agent.annotation.llm_invoke import invoke_json_model
from app.agent.context_helpers import (
    client_active_relative,
    label_names_from_context,
    normalize_rel_path,
    project_context_lines,
    project_directory,
)
from app.schemas.agent import ClientContextInput, TaskIntentLiteral, TurnKindLiteral

logger = logging.getLogger(__name__)

# 回合理解 LLM 的系统提示词（决策规则 + JSON 契约）
UNDERSTAND_SYSTEM = """你是 LR-Agent 回合理解与路由模块。根据 human 消息中的对话历史、界面状态与当前用户输入，一次性输出 JSON（勿分步、勿输出 Markdown）。

## 决策任务

1. 指代消解
   - 将省略/指代结合对话历史、active_relative_path、候选路径，解析为具体 relative_path。
   - referenced_relative_paths：本轮涉及的全部图片路径（可多个）。
   - resolved_active_relative_path：当前主图；无明确主图则为 null。
   - 不得虚构历史中或候选列表中未出现过的路径。

2. 路由 turn_kind（不受 agent_mode 影响）
   - execute_batch：需要启动标注流水线（检测、批量标注、补标、纠正遗漏等）。
   - mutate_annotation：修改或删除已有标注（改标签、删框、批量纠正标签；不含新增检测框）。
   - analyze_data：对标注或项目数据进行统计、分布、聚合分析（将执行 Python 脚本）。
   - generate_report：生成数据分析或标注质量 Markdown 报告。
   - generate_document：生成项目说明、标注规范等 Markdown 文档。
   - converse：问答、解释、查已有标注 JSON、寒暄、需看图描述。
   - clarify_scope：与标注相关但图片范围仍不明确，需先追问。
   - unsupported：当前项目类型无法执行（极少）。

3. 视觉判定 needs_vision_input
   - true：必须分析图像像素才能回答，且读取已有标注 JSON 无法解决。
   - false：读标注文件/文本即可，或无需看图。

## 输出 JSON 结构（必须严格遵守类型）

{
  "resolved_user_content": "补全指代后的完整中文句",
  "referenced_relative_paths": ["data/2.jpg"],
  "resolved_active_relative_path": "data/2.jpg",
  "task_intent": "mutate_annotation",
  "turn_kind": "mutate_annotation",
  "needs_vision_input": false,
  "confidence": 0.9,
  "scope_notes": "",
  "reason": "用户要求删除指定图片的全部标注",
  "user_visible_hint": null
}

## 字段说明

- resolved_user_content：补全指代后的完整中文句，可独立理解。
- task_intent 与 turn_kind 保持一致；查标注 JSON 用 query_annotation，其余与 turn_kind 对齐。
- confidence：0~1 浮点数。
- reason：路由决策依据，**必填字符串**（不可为 null）。
- scope_notes：路径/范围解析的补充说明；无补充时写 **空字符串 \"\"**，**禁止 null**。
- user_visible_hint：仅 turn_kind=clarify_scope 且需向用户追问时填写字符串，否则 **null**。
- resolved_active_relative_path：无明确主图时为 **null**（仅此字段与 user_visible_hint 可为 null）。

所有字符串字段（含 scope_notes、reason）无内容时必须输出 \"\"，不要用 null 表示「不适用」。"""


class TurnUnderstandingLlmResult(BaseModel):
    """LLM 结构化输出的原始解析模型。"""
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

    @field_validator("scope_notes", "reason", mode="before")
    @classmethod
    def _coerce_null_strings(cls, value: object) -> str:
        if value is None:
            return ""
        if isinstance(value, str):
            return value
        return str(value)


class TurnUnderstandingResult(BaseModel):
    """规范化后的回合理解结果，供 orchestrator / assist 下游消费。"""
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
    {
        "execute_batch",
        "mutate_annotation",
        "analyze_data",
        "generate_report",
        "generate_document",
        "converse",
        "clarify_scope",
        "unsupported",
        "wants_batch",
    },
)


def _normalize_turn_kind(raw: str) -> TurnKindLiteral:
    """兼容旧值 wants_batch，非法值回退为 converse。"""
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
    """组装回合理解 LLM 的 human 消息：界面状态 + 对话历史 + 当前用户输入。"""
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
    """将 LLM 原始输出规范化为下游可用的 TurnUnderstandingResult。"""
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
        scope_notes=parsed.scope_notes or "",
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
    """单次 LLM 调用：指代消解 + 意图识别 + 路由决策。"""
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
        log_label="turn_understand",
        null_string_fields=("scope_notes", "reason"),
    )
    result = _llm_to_result(parsed)
    logger.info(
        "[turn_understand] turn_kind=%s task_intent=%s paths=%s scope_notes=%r reason=%r",
        result.turn_kind,
        result.task_intent,
        result.referenced_relative_paths,
        result.scope_notes,
        result.reason,
    )
    return result


def format_understanding_for_system_prompt(result: TurnUnderstandingResult) -> str:
    """将理解结果格式化为【回合理解】块，追加到对话 Agent 系统提示词。"""
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
