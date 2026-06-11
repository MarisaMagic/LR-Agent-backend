"""Agent Markdown 报告/文档生成（单次 LLM，基于数据快照）。"""

from __future__ import annotations

import json
from typing import Literal

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field, field_validator

from app.agent.annotation.llm_invoke import invoke_json_model
from app.agent.text_sanitize import sanitize_unicode_text

ReportKind = Literal["report", "document"]

REPORT_PREPARE_SYSTEM = """你是 LR-Agent 报告与文档生成助手。根据对话上下文、用户请求与标注项目数据快照，撰写完整 Markdown。

输出 JSON（一次完成，勿用 markdown 代码块包裹 JSON）：
{
  "title": "报告标题",
  "content": "# 标题\\n\\n## 摘要\\n...完整 Markdown 正文...",
  "suggested_relative_path": "reports/annotation-quality-20250611.md",
  "summary": "一两句话说明本报告内容"
}

规则：
- content 必须是完整 Markdown 正文；禁止输出 {"name":"generate_report"} 等伪工具 JSON
- 统计数字必须来自数据快照，勿编造
- suggested_relative_path：相对项目根，仅 .md，建议放在 reports/ 目录
- report_kind=report：数据分析或标注质量报告（摘要、统计、发现、建议）
- report_kind=document：项目说明、标注规范或使用文档
- title、summary 为简短中文
"""


class ReportPrepareLlmResult(BaseModel):
    title: str = Field(min_length=1, max_length=500)
    content: str = Field(min_length=1, max_length=120_000)
    suggested_relative_path: str = Field(min_length=1, max_length=500)
    summary: str = ""

    @field_validator("title", "content", "suggested_relative_path", "summary", mode="before")
    @classmethod
    def _coerce_null_strings(cls, value: object) -> str:
        if value is None:
            return ""
        if isinstance(value, str):
            return value
        return str(value)


class ReportPrepareResult(BaseModel):
    title: str
    content: str
    suggested_relative_path: str
    summary: str


def _normalize_relative_path(path: str, *, report_kind: ReportKind) -> str:
    cleaned = path.strip().replace("\\", "/").lstrip("/")
    if not cleaned.endswith(".md"):
        cleaned = f"{cleaned}.md" if cleaned else "reports/output.md"
    if not cleaned.lower().startswith("reports/"):
        cleaned = f"reports/{cleaned}"
    if report_kind == "document" and "reports/" in cleaned and "docs/" not in cleaned:
        cleaned = cleaned.replace("reports/", "docs/", 1)
    return cleaned


async def prepare_report_markdown(
    llm: ChatOpenAI,
    *,
    user_request: str,
    data_snapshot: dict,
    report_kind: ReportKind = "report",
    conversation_transcript: str = "",
) -> ReportPrepareResult:
    snapshot_text = json.dumps(data_snapshot, ensure_ascii=False)[:120_000]
    transcript = conversation_transcript.strip() or "（无历史）"
    human = (
        f"report_kind: {report_kind}\n\n"
        f"【对话上下文】\n{transcript}\n\n"
        f"【用户请求】\n{user_request.strip()}\n\n"
        f"【数据快照 JSON】\n{snapshot_text}\n"
    )
    parsed = await invoke_json_model(
        llm,
        messages=[
            SystemMessage(content=REPORT_PREPARE_SYSTEM),
            HumanMessage(content=human),
        ],
        model_cls=ReportPrepareLlmResult,
        log_label="report_prepare",
        null_string_fields=("title", "content", "suggested_relative_path", "summary"),
    )
    title = sanitize_unicode_text(parsed.title.strip()) or "数据报告"
    content = sanitize_unicode_text(parsed.content.strip())
    summary = sanitize_unicode_text(parsed.summary.strip())
    rel = _normalize_relative_path(
        parsed.suggested_relative_path.strip() or "reports/annotation-report.md",
        report_kind=report_kind,
    )
    return ReportPrepareResult(
        title=title,
        content=content,
        suggested_relative_path=rel,
        summary=summary or title,
    )
