"""标注变更准备：单次 LLM 解析 mutate 意图与目标描述。"""

from __future__ import annotations

import json
from typing import Any, Literal

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

from app.agent.annotation.debug_log import log_annotation_agent
from app.agent.annotation.json_utils import extract_json_object
from app.agent.annotation.llm_invoke import invoke_json_model

MUTATION_PREPARE_SYSTEM = """你是 LR-Agent 标注变更准备助手。用户希望修改或删除已有标注（非新增检测框）。

输出 JSON（一次完成）：
{
  "selected_paths": ["data/7.jpg"],
  "intent_summary": "一句话摘要",
  "operations": [
    {
      "relative_path": "data/7.jpg",
      "mutation_kind": "patch_label|delete",
      "targets": [
        {"by": "all"},
        {"by": "id", "id": "uuid"},
        {"by": "label_name", "label_name": "person"},
        {"by": "index", "index": 1},
        {"by": "spatial", "hint": "leftmost"}
      ],
      "new_label_name": "worker"
    }
  ]
}

规则：
- mutation_kind=patch_label 时必须给出 new_label_name（项目标签名之一）
- mutation_kind=delete 时不需要 new_label_name
- selected_paths 必须为候选列表中的 relative_path
- targets 描述如何定位框；优先使用 id；无 id 时用 label_name/index/spatial
- 用户要求删除/清空某张图的全部标注时：mutation_kind=delete，targets=[{"by":"all"}]
- 用户说「这个框」「当前选中」时，在 targets 中加入 {"by":"selected"}
- 勿编造不在候选中的路径
"""


class MutationTargetSchema(BaseModel):
    by: Literal["id", "label_name", "index", "spatial", "selected", "all"] = "id"
    id: str | None = None
    label_name: str | None = None
    index: int | None = None
    hint: str | None = None


class MutationOperationSchema(BaseModel):
    relative_path: str = Field(min_length=1)
    mutation_kind: Literal["patch_label", "delete"] = "patch_label"
    targets: list[MutationTargetSchema] = Field(default_factory=list)
    new_label_name: str | None = None


class MutationPrepareLlmResult(BaseModel):
    selected_paths: list[str] = Field(default_factory=list)
    intent_summary: str = ""
    operations: list[MutationOperationSchema] = Field(default_factory=list)


class MutationPrepareResult(BaseModel):
    selected_paths: list[str] = Field(default_factory=list)
    intent_summary: str = ""
    operations: list[dict[str, Any]] = Field(default_factory=list)
    resolved_user_request: str = ""


def _filter_paths(raw_paths: object, candidates: list[dict]) -> list[str]:
    by_path = {
        str(c.get("relative_path") or "").strip().replace("\\", "/"): c
        for c in candidates
        if str(c.get("relative_path") or "").strip()
    }
    selected: list[str] = []
    if isinstance(raw_paths, list):
        for raw in raw_paths:
            key = str(raw).strip().replace("\\", "/")
            if key in by_path:
                selected.append(key)
    return selected


async def prepare_mutation_annotation(
    llm: ChatOpenAI,
    *,
    user_request: str,
    current_relative_path: str,
    candidates: list[dict],
    label_names: list[str],
    selected_annotation_ids: list[str] | None = None,
    conversation_transcript: str = "",
) -> MutationPrepareResult:
    if not candidates:
        return MutationPrepareResult(
            intent_summary="无图片候选",
            resolved_user_request=user_request,
        )

    catalog = json.dumps(
        [
            {
                "relative_path": c.get("relative_path"),
                "name": c.get("name"),
                "parent": c.get("parent"),
            }
            for c in candidates[:80]
        ],
        ensure_ascii=False,
    )
    labels = ", ".join(label_names[:50]) or "（无）"
    selected_ids = ", ".join(selected_annotation_ids or []) or "（无）"
    transcript = conversation_transcript.strip() or "（无历史）"

    human = (
        f"【对话上下文】\n{transcript}\n\n"
        f"【用户请求】\n{user_request.strip()}\n\n"
        f"current_relative_path: {current_relative_path or '（无）'}\n"
        f"selected_annotation_ids: {selected_ids}\n"
        f"项目标签: {labels}\n"
        f"候选图片: {catalog}\n"
    )

    parsed = await invoke_json_model(
        llm,
        messages=[
            SystemMessage(content=MUTATION_PREPARE_SYSTEM),
            HumanMessage(content=human),
        ],
        model_cls=MutationPrepareLlmResult,
    )

    selected = _filter_paths(parsed.selected_paths, candidates)
    if not selected and parsed.operations:
        op_paths = {
            str(op.relative_path).strip().replace("\\", "/")
            for op in parsed.operations
        }
        selected = [p for p in op_paths if p in {c.get("relative_path") for c in candidates}]

    operations = [op.model_dump() for op in parsed.operations]

    log_annotation_agent(
        "mutation-prepare",
        "mutation-prepare 完成",
        selected=len(selected),
        operations=len(operations),
    )

    return MutationPrepareResult(
        selected_paths=selected,
        intent_summary=parsed.intent_summary or user_request[:200],
        operations=operations,
        resolved_user_request=user_request,
    )
