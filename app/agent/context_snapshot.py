"""Assist 模式系统提示词组装：运行时身份、任务指令与客户端上下文快照。"""
from __future__ import annotations

from app.schemas.agent import AnnotationProjectSnapshotInput, ClientContextInput, SkillCatalogEntryInput

# 精简后的 prompt 常量 —— 规则由代码层承担，不靠模型理解
VISION_HINT = """- 看图描述：按需调用 read_image_for_vision。
- 查已有标注 JSON：read_file_annotation。"""

# 不同标注类型的工具使用指导
ANNOTATION_TOOL_GUIDE = {
    "bbox": "【标注工具】使用 auto_annotate 执行自动标注。scope_hint 从目录浏览结果中取 relativePath 值。",
    "caption": "【标注工具】使用 auto_annotate 执行自动标注。scope_hint 从目录浏览结果中取 relativePath 值。",
    "classification": "【标注工具】使用 auto_annotate 执行自动标注。scope_hint 从目录浏览结果中取 relativePath 值。",
    "polygon": "【标注工具】使用 auto_annotate 执行自动标注。scope_hint 从目录浏览结果中取 relativePath 值。需配置检测模型与 SAM2 分割模型。",
    "keypoint": "【标注工具】使用 auto_annotate 执行自动标注。scope_hint 从目录浏览结果中取 relativePath 值。需先选择骨架模板并配置关键点模型。",
    "rotated_bbox": "【标注工具】使用 auto_annotate 执行自动标注。scope_hint 从目录浏览结果中取 relativePath 值。需配置 OBB 检测模型。",
    "span_ner": "【标注工具】使用 auto_annotate 执行自动标注。scope_hint 从目录浏览结果中取 relativePath 值。",
    "text_classification": "【标注工具】使用 auto_annotate 执行自动标注。scope_hint 从目录浏览结果中取 relativePath 值。",
    "instruction": "【标注工具】使用 auto_annotate 执行自动标注。scope_hint 从目录浏览结果中取 relativePath 值。",
    "preference": "【标注工具】使用 auto_annotate 执行自动标注。scope_hint 从目录浏览结果中取 relativePath 值。",
    "conversation": "【标注工具】使用 auto_annotate 执行自动标注。scope_hint 从目录浏览结果中取 relativePath 值。",
    "cot": "【标注工具】使用 auto_annotate 执行自动标注。scope_hint 从目录浏览结果中取 relativePath 值。",
}

# 不需要展示检测模型列表的标注类型
_DETECTION_MODEL_TYPES: frozenset[str] = frozenset({"bbox", "polygon", "keypoint", "rotated_bbox"})

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
    annotation_type = snapshot.annotation_type or ""
    lines = [
        f"项目 ID: {snapshot.project_id}",
        f"项目名称: {snapshot.name}",
        f"模态: {snapshot.modality}",
        f"标注类型: {annotation_type}",
        "标签列表:",
        *(label_lines or ["- （无）"]),
    ]

    # 仅在目标检测类标注类型时展示可用的检测模型
    if annotation_type in _DETECTION_MODEL_TYPES:
        models = snapshot.detection_models or []
        model_lines = [
            f"- {m.get('id', '')}: {m.get('name', '')}"
            + (" (默认)" if m.get("is_default") else "")
            for m in models[:12]
            if isinstance(m, dict)
        ]
        lines.append("可用检测模型（object_detection）:")
        lines.extend(model_lines or ["- （未配置或未传入）"])

    # 追加工具使用指导
    tool_guide = ANNOTATION_TOOL_GUIDE.get(annotation_type)
    if tool_guide:
        lines.append(tool_guide)

    return "\n".join(lines)


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


def format_skills_catalog_block(
    skills: list[SkillCatalogEntryInput] | None,
) -> str:
    """把全局 skills catalog（name + description）格式化为 prompt 块；空时不输出。"""
    if not skills:
        return ""
    lines = [
        "【可用 Skills】",
        "以下为可用的任务工作流 Skills。当用户请求与某 skill 的 description 匹配时，"
        "先调用 read_agent_skill(skill_name) 读取该 SKILL.md 正文，再按其中步骤执行；"
        "不要凭名字猜测内容。",
    ]
    for item in skills:
        name = (item.name or "").strip()
        desc = (item.description or "").strip()
        if not name or not desc:
            continue
        lines.append(f"- {name}: {desc}")
    return "\n".join(lines)


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
    editor_note = ""
    if client_context and client_context.work_mode == "editor":
        editor_note = (
            "\n【编辑器模式】当前为编辑器模式：禁止调用标注读写、批量标注、标注变更与标注数据分析相关工具；"
            "可使用 read_workspace_file、write_workspace_file、read_document_file 等通用工具。"
        )
    base = f"{identity}\n\n{task}"
    if editor_note:
        base = f"{base}{editor_note}"
    instructions = (client_context.project_instructions or "").strip() if client_context else ""
    if instructions:
        base = f"{base}\n\n【项目指令】\n{instructions}"
    memory_index = (client_context.memory_index or "").strip() if client_context else ""
    if memory_index:
        base = (
            f"{base}\n\n【已保存的记忆】\n{memory_index}\n"
            "（以上为跨会话记忆索引；需要细节时用 memory_read 读取对应 topic 文件。"
            "当用户明确要求「记住」某事，或纠正了你的做法且该纠正具有长期价值时，"
            "用 memory_write 保存简洁的 markdown 记忆并附索引行。项目指令优先级高于记忆。）"
        )
    skills = (client_context.skills_catalog or []) if client_context else []
    skills_block = format_skills_catalog_block(skills)
    if skills_block:
        base = f"{base}\n\n{skills_block}"
    return base
