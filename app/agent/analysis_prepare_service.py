"""Agent 数据分析脚本生成（含校验与 repair 重试）。"""

from __future__ import annotations

import json
import re
import textwrap
from typing import Literal

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

from app.agent.analysis_script_validate import ScriptGuardError, validate_script
from app.agent.annotation.llm_invoke import invoke_json_model
from app.agent.text_sanitize import sanitize_unicode_text

ANALYSIS_PREPARE_SYSTEM = """你是 LR-Agent 数据分析脚本生成器。根据对话上下文、用户请求与数据快照，输出可在沙箱执行的 Python 脚本。

## 可用数据

- 沙箱执行前注入全局变量 DATA（dict），内容与下方【数据快照 JSON】一致
- DATA 顶层字段：projectId, projectName, totalFiles, annotatedFiles, totalBoxes
- labelCounts: { "标签名": 数量 }
- files: [ { relativePath, boxCount, labelCounts } ]
- 工作目录另有 annotations.json（与 DATA 相同），仅作无 DATA 时的备选

## 写法要求

- 编写模块顶层语句，不要整体多缩进一层，不要把整段包进无意义的 try/except
- 用 data = DATA 读取快照（DATA 一定存在，无需 try/except NameError）
- 用 print() 输出分析结果；结构化结果可用 print(json.dumps(..., ensure_ascii=False))
- 可用标准库：json, csv, statistics, collections, datetime, math, re, itertools, functools, decimal, pathlib, operator, copy, heapq, bisect, textwrap, string, enum, fractions
- 禁止 open()、os、sys、subprocess、socket 等；优先 DATA，少用 Path.read_text

## 输出格式

只输出一个 JSON 对象（不要 markdown 代码块）：
{"script": "完整 Python 脚本", "explanation": "一句话说明脚本做什么（中文）", "input_files": ["annotations.json"]}
"""


class AnalysisRepairContext(BaseModel):
    previous_script: str = ""
    error_message: str = Field(min_length=1, max_length=8_000)
    error_stage: Literal["syntax", "guard", "runtime"] = "runtime"
    stdout: str | None = Field(default=None, max_length=32_000)


class AnalysisPrepareLlmResult(BaseModel):
    script: str = Field(min_length=1, max_length=32_000)
    explanation: str = ""
    input_files: list[str] = Field(default_factory=list)


class AnalysisPrepareResult(BaseModel):
    script: str
    explanation: str
    input_files: list[str] = Field(default_factory=list)
    attempts: int = 1


_OPEN_CALL_PATTERN = re.compile(r"\bopen\s*\(")
_MAX_VALIDATION_ATTEMPTS = 3


def _line_indent(line: str) -> int:
    return len(line) - len(line.lstrip(" \t"))


def _normalize_analysis_script(script: str) -> str:
    cleaned = sanitize_unicode_text(script.strip())
    if not cleaned:
        return ""
    cleaned = textwrap.dedent(f"\n{cleaned}").lstrip("\n").strip()
    lines = cleaned.splitlines()
    if not lines:
        return cleaned
    indents = [_line_indent(line) for line in lines if line.strip()]
    if not indents:
        return cleaned
    if indents[0] == 0:
        body_indents = [i for i in indents[1:] if i > 0]
        if body_indents and len(set(body_indents)) == 1:
            pad = body_indents[0]
            lines = [
                lines[0],
                *[
                    line[pad:] if line.strip() and _line_indent(line) >= pad else line
                    for line in lines[1:]
                ],
            ]
            cleaned = "\n".join(lines)
    else:
        min_indent = min(indents)
        if min_indent > 0:
            lines = [
                line[min_indent:] if line.strip() else line
                for line in lines
            ]
            cleaned = "\n".join(lines)
    return cleaned.strip()


def _validate_analysis_script(script: str) -> str | None:
    """返回 None 表示通过，否则返回错误信息。"""
    if _OPEN_CALL_PATTERN.search(script):
        return "脚本包含禁止的 open()，请改用 DATA 或 Path.read_text"
    try:
        validate_script(script)
    except ScriptGuardError as exc:
        return str(exc)
    return None


def _build_human_message(
    *,
    user_request: str,
    data_snapshot: dict,
    conversation_transcript: str,
    repair_context: AnalysisRepairContext | None,
) -> str:
    snapshot_text = json.dumps(data_snapshot, ensure_ascii=False)[:120_000]
    transcript = conversation_transcript.strip() or "（无历史）"
    parts = [
        f"【对话上下文】\n{transcript}\n",
        f"【用户请求】\n{user_request.strip()}\n",
        f"【数据快照 JSON】\n{snapshot_text}\n",
    ]
    if repair_context is not None:
        stage_label = {
            "syntax": "语法",
            "guard": "沙箱规则",
            "runtime": "运行",
        }.get(repair_context.error_stage, "运行")
        parts.append(
            f"【上次脚本】\n{repair_context.previous_script[:16_000]}\n\n"
            f"【{stage_label}错误】\n{repair_context.error_message.strip()}\n"
        )
        if repair_context.stdout and repair_context.stdout.strip():
            parts.append(f"【部分输出】\n{repair_context.stdout.strip()[:8_000]}\n")
        parts.append("请修复脚本，确保模块顶层无多余缩进、可直接执行。\n")
    return "\n".join(parts)


async def prepare_analysis_script(
    llm: ChatOpenAI,
    *,
    user_request: str,
    data_snapshot: dict,
    conversation_transcript: str = "",
    repair_context: AnalysisRepairContext | None = None,
) -> AnalysisPrepareResult:
    internal_repair = repair_context
    last_parsed: AnalysisPrepareLlmResult | None = None
    script = ""
    attempts = 0

    for attempt in range(_MAX_VALIDATION_ATTEMPTS):
        attempts = attempt + 1
        human = _build_human_message(
            user_request=user_request,
            data_snapshot=data_snapshot,
            conversation_transcript=conversation_transcript,
            repair_context=internal_repair,
        )
        log_label = "analysis_prepare" if attempt == 0 else f"analysis_prepare_repair_{attempt}"
        parsed = await invoke_json_model(
            llm,
            messages=[
                SystemMessage(content=ANALYSIS_PREPARE_SYSTEM),
                HumanMessage(content=human),
            ],
            model_cls=AnalysisPrepareLlmResult,
            log_label=log_label,
            null_string_fields=("explanation",),
        )
        last_parsed = parsed
        script = _normalize_analysis_script(parsed.script)
        error = _validate_analysis_script(script)
        if error is None:
            return AnalysisPrepareResult(
                script=script,
                explanation=sanitize_unicode_text(parsed.explanation.strip()),
                input_files=parsed.input_files or ["annotations.json"],
                attempts=attempts,
            )
        internal_repair = AnalysisRepairContext(
            previous_script=script,
            error_message=error,
            error_stage="guard" if "禁止" in error or "不允许" in error else "syntax",
        )

    assert last_parsed is not None
    raise ValueError(
        f"分析脚本校验失败（{attempts} 次）: {internal_repair.error_message if internal_repair else 'unknown'}"
    )
