"""Assist 模式系统提示词组装：运行时身份、任务指令与客户端上下文快照。

由 orchestrator 在 assist 路由下调用，组装顺序：
  1. 运行时身份块（模型 ID、视觉能力探针结果）
  2. 任务指令模板（工具纪律、视觉规则、文件读取指引）
  3. 可选上下文（标注项目快照 / 工作区路径 / 当前打开文件）
"""

from __future__ import annotations

from app.schemas.agent import AnnotationProjectSnapshotInput, ClientContextInput

# 工具调用行为规范：禁止口述「准备调工具」，直接发起 tool call
TOOL_DISCIPLINE = """【工具与回复顺序】
- 需要工具时：直接发起 tool call，禁止先输出「请稍等」「我将调用…」等口述；最终答案在工具完成（及可能的附图注入）之后输出。
- 用户只会看到：工具块（可折叠）→ 你的正文回答；不要把「准备调工具」当作最终回复。"""

# 视觉与标注工具的选择规则
VISION_RULES = """【图片与标注】
- 用户问图中视觉内容（人数、物体、场景、文字 OCR、外观、「是谁」且需看像素等）：使用 `read_image_for_vision`；成功后系统注入附图，再根据图像回答。不要用 `read_file_annotation` 代替。
- 仅当用户明确问「标注 JSON/文件里标了谁、有哪些框」时：使用 `read_file_annotation`。
- `read_image_for_vision` 的 relative_path 为空时使用当前打开的图片；用户给出如 data/1.jpg 时传入该相对路径。"""

# 仅有工作区、无标注项目时的任务模板
WORKSPACE_ASSIST_TASK = """【任务】在 LR-Agent 系统内回答用户问题、按工具结果执行读文件等操作。
""" + TOOL_DISCIPLINE + """
- 文本/代码： `read_workspace_file`
- PDF/DOCX： `read_document_file`
""" + VISION_RULES + """
- 路径参数为空时优先使用当前打开的文件
- 用自然、简洁的中文回复；禁止在回复中带 [Ask]、[Agent] 等前缀

可使用工具查询账户、应用说明与界面上下文。"""

# 已打开标注项目时的任务模板（含批量检测路由约束）
ASSISTANT_TASK_BASE = """【任务】在 LR-Agent 系统内回答用户问题、按工具与路由执行相关操作；当前已打开标注项目。
""" + TOOL_DISCIPLINE + """
- 标注策略、标签含义、项目结构：结合项目快照与工具回答
""" + VISION_RULES + """
- 文本/代码： `read_workspace_file`；PDF/DOCX： `read_document_file`
- 【回合理解】路由为 execute_batch 时，批量检测与写入由客户端执行；本条对话中勿声称已完成检测或写入标注文件
- 用自然、简洁的中文回复；禁止在回复中带 [Ask]、[Agent]、「助手:」等前缀

可使用工具查询账户、应用说明、界面上下文、文件内容与单文件标注。"""


def format_runtime_identity_block(
    *,
    model: str,
    provider_label: str = "",
    supports_vision: bool = False,
) -> str:
    """生成【你的身份】块：告知模型自身 ID、视觉能力探针结果与系统边界。"""
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
    """将标注项目快照格式化为提示词可读文本（标签、检测模型等）。"""
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
    """工作区 assist 提示词：任务模板 + 工作区根目录与当前打开文件。"""
    if client_context is None:
        return WORKSPACE_ASSIST_TASK
    parts = [WORKSPACE_ASSIST_TASK]
    if client_context.workspace_root:
        parts.append(f"\n【工作区】\n根目录: {client_context.workspace_root}")
    if client_context.active_file_path:
        parts.append(f"当前打开文件: {client_context.active_file_path}")
    return "\n".join(parts)


def build_project_assistant_system_prompt(client_context: ClientContextInput | None) -> str:
    """标注项目 assist 提示词：任务模板 + 项目快照。"""
    if client_context is None or client_context.annotation_project_snapshot is None:
        return ASSISTANT_TASK_BASE
    snap = client_context.annotation_project_snapshot
    return (
        f"{ASSISTANT_TASK_BASE}\n\n"
        f"【当前标注项目快照】\n{format_snapshot_for_prompt(snap)}"
    )


def build_assist_system_prompt(
    client_context: ClientContextInput | None,
    *,
    model: str,
    provider_label: str = "",
    supports_vision: bool = False,
) -> str:
    """组装完整 assist 系统提示词：身份块 + 按上下文选择任务模板。"""
    identity = format_runtime_identity_block(
        model=model,
        provider_label=provider_label,
        supports_vision=supports_vision,
    )
    if client_context and client_context.annotation_project_snapshot is not None:
        task = build_project_assistant_system_prompt(client_context)
    elif client_context and (client_context.workspace_root or "").strip():
        task = build_workspace_assistant_system_prompt(client_context)
    else:
        task = WORKSPACE_ASSIST_TASK
    return f"{identity}\n\n{task}"
