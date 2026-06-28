"""Assist 模式系统提示词组装：运行时身份、任务指令与客户端上下文快照。"""

from __future__ import annotations

from app.schemas.agent import AnnotationProjectSnapshotInput, ClientContextInput

WRITE_TOOL_GUIDE = """【写文件工具 write_workspace_file】
- 创建/覆写报告、文档、代码、配置：必须调用 `write_workspace_file`（relative_path + content）。
- 支持常见文本与代码后缀（.md .txt .json .yaml .py .cpp .ts 等）；父目录不存在时会自动创建。
- 调用成功后仅生成 **待确认提案**；用户点击 Keep All 后才会写入磁盘。
- **严禁幻觉**：未成功调用 write_workspace_file，或 ToolMessage 未返回提案成功信息时，禁止声称「已保存」「已写入」「文件在 xxx 路径」。
- 复合任务（如先分析/标注再写报告）：客户端工具完成后，若仍需落盘，必须继续调用 write_workspace_file，然后再做文字总结。"""

TOOL_DISCIPLINE = """【工具与回复顺序】
- 需要工具时：直接发起 tool call，禁止先输出「请稍等」「我将调用…」等口述。
- 禁止在正文中写 `execute_batch_annotation(...)` 等伪代码；必须发起真实 tool call。
- 根据用户任务**自行选择**合适工具：写文件/代码用 write_workspace_file；批量标注用 execute_batch_annotation；改标用 mutate_annotation；统计分析用 analyze_data。
- 不要在不相关的任务上调用 execute_batch_annotation（例如写报告、写算法代码、创建文件夹）。"""

VISION_RULES = """【图片与标注】
- 看图描述（人数、物体、OCR 等）：`read_image_for_vision`。
- 查已有标注 JSON：`read_file_annotation`。
- 批量检测标注：`execute_batch_annotation`（仅当用户确实需要标注图片时）。"""

EXECUTION_TOOL_GUIDE = """【客户端执行工具】（仅在与用户任务匹配时调用）
- execute_batch_annotation：批量检测并标注图片；user_request 传用户原话。
- mutate_annotation：修改/删除已有标注框。
- analyze_data：对标注数据统计分析。
- 以上工具与 write_workspace_file 可串联；全部必要步骤完成后再写最终总结。"""

PLANNING_GUIDE = """【任务规划】
- 复合任务可在 reasoning 中列简要步骤，然后对第一步发起真实 tool call。
- 每一步等 ToolMessage 返回后再决定下一步；需要落盘时不得跳过 write_workspace_file。"""

FILE_WRITE_INTEGRITY = """【文件写入诚信】
- 只有 write_workspace_file 的 ToolMessage 明确表示提案已生成时，才可告知用户「请在上方变更列表中 Keep All 确认写入」。
- 禁止编造文件路径、禁止展示「已写入磁盘」的虚假结论。"""

WORKSPACE_ASSIST_TASK = """【任务】在 LR-Agent 内回答用户问题，并用工具完成读/写文件等操作。
""" + TOOL_DISCIPLINE + """
- 读文本/代码：`read_workspace_file`；PDF/DOCX：`read_document_file`
""" + VISION_RULES + """
""" + WRITE_TOOL_GUIDE + """
""" + FILE_WRITE_INTEGRITY + """
- 用自然、简洁的中文回复。"""

ASSISTANT_TASK_BASE = """【任务】在 LR-Agent 内完成问答、标注、分析、写文件等任务。
""" + TOOL_DISCIPLINE + """
""" + PLANNING_GUIDE + """
""" + VISION_RULES + """
- 读文本/代码：`read_workspace_file`；PDF/DOCX：`read_document_file`
""" + EXECUTION_TOOL_GUIDE + """
""" + WRITE_TOOL_GUIDE + """
""" + FILE_WRITE_INTEGRITY + """
- 用自然、简洁的中文回复。"""


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
    if client_context is None:
        return WORKSPACE_ASSIST_TASK
    parts = [WORKSPACE_ASSIST_TASK]
    if client_context.workspace_root:
        parts.append(f"\n【工作区】\n根目录: {client_context.workspace_root}")
    if client_context.active_file_path:
        parts.append(f"当前打开文件: {client_context.active_file_path}")
    return "\n".join(parts)


def build_project_assistant_system_prompt(client_context: ClientContextInput | None) -> str:
    if client_context is None or client_context.annotation_project_snapshot is None:
        return ASSISTANT_TASK_BASE
    snap = client_context.annotation_project_snapshot
    return (
        f"{ASSISTANT_TASK_BASE}\n\n"
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
