"""Format annotation project snapshot for Ask-mode system prompts."""
from __future__ import annotations

import json
from typing import Any

from app.schemas.agent import AnnotationProjectSnapshotInput, ClientContextInput

ASK_SYSTEM_BASE = """你正在 LR-Agent 的 **Ask 模式** 中协助用户。

职责：聊天、分析、解读标签与标注策略、说明目录与任务类型、回答「如何标注」类问题。
**不要**假装已经执行了批量检测或写入了标注文件；执行标注请提示用户切换到 **Agent（Annotation）模式**。

可使用工具查询账户、应用说明与当前标注项目上下文。"""


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


def build_ask_system_prompt(client_context: ClientContextInput | None) -> str:
    if client_context is None or client_context.annotation_project_snapshot is None:
        return ASK_SYSTEM_BASE
    snap = client_context.annotation_project_snapshot
    mode_line = (
        "当前为 Agent 执行模式请求，但本条走对话流；仅作上下文参考。"
        if client_context.agent_mode in ("annotation", "annotate")
        else "当前客户端为 Chat（Ask）模式。"
    )
    return (
        f"{ASK_SYSTEM_BASE}\n\n{mode_line}\n\n"
        f"【当前标注项目快照】\n{format_snapshot_for_prompt(snap)}"
    )
