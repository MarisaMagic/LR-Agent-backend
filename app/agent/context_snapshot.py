"""Assist 模式系统提示词组装：运行时身份、任务指令与客户端上下文快照。

约束规则已移至 ToolGuard（程序化）和 AssistModeRouter（工具集裁剪），
prompt 仅保留身份 + 任务简述，不再包含大段工具纪律文本。
"""

from __future__ import annotations

from app.schemas.agent import AnnotationProjectSnapshotInput, ClientContextInput

# 精简后的 prompt 常量 —— 规则由代码层承担，不靠模型理解
VISION_HINT = """- 看图描述：按需调用 read_image_for_vision。
- 查已有标注 JSON：read_file_annotation。"""

WORKSPACE_ASSIST_TASK = """【任务】在 LR-Agent 内回答用户问题，按需使用工具完成读/写/分析操作。
{vision_hint}
- 用自然、简洁的中文回复。"""

ASSISTANT_TASK_BASE = """【任务】在 LR-Agent 内完成问答、标注、分析、写文件等任务。
按需使用工具；系统自动管理文件写入确认与工具调度。
{vision_hint}
- 用自然、简洁的中文回复。"""

# 向后兼容保留旧常量引用（如有外部引用）
WRITE_TOOL_GUIDE = ""
TOOL_DISCIPLINE = ""
VISION_RULES = ""
EXECUTION_TOOL_GUIDE = ""
PLANNING_GUIDE = ""
FILE_WRITE_INTEGRITY = ""


def format_runtime_identity_block(
    *,
    model: str,
    provider_label: str = "",
    supports_vision: bool = False,
) -> str:
    label = provider_label.strip() or "（未命名提供商）"
    if supports_vision:
        vision_line = "视觉能力：已通过 API 探针，可调用 read_image_for_vision 并在调用后查看附图。"
    else:
        vision_line = (
            "视觉能力：未通过探针或未检测。不要声称能分析图片像素；"
            "若用户要看图，说明需在「大模型配置」中选用支持视觉的模型并重新检测视觉。"
        )
    return (
        "【你的身份】\n"
        f"- 你是后端模型 `{model}`（配置名称：{label}）。\n"
        f"- {vision_line}\n"
        "- 你在 LR-Agent 系统内与用户对话、调用工具完成任务；不要把自己说成独立的「LR-Agent 助手」或其它品牌模型。"
    )


def format_snapshot_for_prompt(snapshot: AnnotationProjectSnapshotInput) -> str:
    labels = snapshot.labels or []
    label_lines = [
        f"- {item.get('id', '')}: {item.get('name', '')}"
        for item in labels[:40]
        if isinstance(item, dict)
    ]
    models = snapshot.detection_models or []
    model_lines = [
        f"- {m.get('id', '')}: {m.get('name', '')}"
        + (" (默认)" if m.get("is_default") else "")
        for m in models[:12]
        if isinstance(m, dict)
    ]
    return "\n".join(
        [
            f"项目 ID: {snapshot.project_id}",
            f"项目名称: {snapshot.name}",
            f"模态: {snapshot.modality}",
            f"标注类型: {snapshot.annotation_type}",
            "标签列表:",
            *(label_lines or ["- （无）"]),
            "可用检测模型（object_detection）:",
            *(model_lines or ["- （未配置或未传入）"]),
        ]
    )


def build_workspace_assistant_system_prompt(client_context: ClientContextInput | None) -> str:
    task = WORKSPACE_ASSIST_TASK.format(vision_hint=VISION_HINT)
    if client_context is None:
        return task
    parts = [task]
    if client_context.workspace_root:
        parts.append(f"\n【工作区】\n根目录: {client_context.workspace_root}")
    if client_context.active_file_path:
        parts.append(f"当前打开文件: {client_context.active_file_path}")
    return "\n".join(parts)


def build_project_assistant_system_prompt(client_context: ClientContextInput | None) -> str:
    task = ASSISTANT_TASK_BASE.format(vision_hint=VISION_HINT)
    if client_context is None or client_context.annotation_project_snapshot is None:
        return task
    snap = client_context.annotation_project_snapshot
    return (
        f"{task}\n\n"
        f"【当前标注项目快照】\n{format_snapshot_for_prompt(snap)}"
    )


def format_turn_task_addon(client_context: ClientContextInput | None) -> str:
    """注入回合理解 LLM 的路由结论（reason / turn_kind），非用户关键词规则。"""
    if client_context is None or client_context.turn_understanding is None:
        return ""
    tu = client_context.turn_understanding
    parts: list[str] = []
    if (tu.reason or "").strip():
        parts.append(f"路由依据：{tu.reason.strip()}")
    if (tu.turn_kind or "").strip():
        parts.append(f"turn_kind：{tu.turn_kind.strip()}")
    if (tu.scope_notes or "").strip():
        parts.append(f"范围：{tu.scope_notes.strip()}")
    if not parts:
        return ""
    return "\n【回合理解】" + "；".join(parts)


def build_assist_system_prompt(
    client_context: ClientContextInput | None,
    *,
    model: str,
    provider_label: str = "",
    supports_vision: bool = False,
) -> str:
    identity = format_runtime_identity_block(
        model=model,
        provider_label=provider_label,
        supports_vision=supports_vision,
    )
    if client_context and client_context.work_mode == "editor":
        task = build_workspace_assistant_system_prompt(client_context)
    elif client_context and client_context.annotation_project_snapshot is not None:
        task = build_project_assistant_system_prompt(client_context)
    elif client_context and (client_context.workspace_root or "").strip():
        task = build_workspace_assistant_system_prompt(client_context)
    else:
        task = WORKSPACE_ASSIST_TASK
    addon = format_turn_task_addon(client_context)
    editor_note = ""
    if client_context and client_context.work_mode == "editor":
        editor_note = (
            "\n【编辑器模式】当前为编辑器模式：禁止调用标注读写、批量标注、标注变更与标注数据分析相关工具；"
            "可使用 read_workspace_file、write_workspace_file、read_document_file 等通用工具。"
        )
    base = f"{identity}\n\n{task}"
    combined = f"{base}\n{addon}" if addon else base
    return f"{combined}{editor_note}" if editor_note else combined
