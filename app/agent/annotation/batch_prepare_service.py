"""批量标注准备：单次 LLM 同时完成图片范围选择与执行计划生成。

替代原先多步 parse-task + parse-scope + create-plan 流程。
由 annotation_agent API `/batch-prepare` 调用，前端在 execute_batch 路由后发起。

输出 BatchPrepareResult：
  - selected_paths：本轮待标注图片相对路径
  - scope_reason：选图依据
  - plan：子 Agent 执行计划（检测参数、标签策略、视觉映射、annotation_scope 等）
"""

from __future__ import annotations

import json

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

from app.agent.annotation.annotation_scope import merge_annotation_scope
from app.agent.annotation.debug_log import log_annotation_agent
from app.agent.annotation.json_utils import extract_json_object
from app.agent.annotation.label_vision_policy import labels_require_vision_mapping
from app.agent.annotation.plan_parsing import build_batch_plan_from_data
from app.agent.annotation.schemas import AnnotationScopePayload, BatchPlanResult, BatchPrepareResult

BATCH_PREPARE_SYSTEM = """你是批量图片 bbox 标注的准备助手（一次输出，勿分步啰嗦）。

你需要同时完成：
1. 从候选图片列表中选出本轮要标注的文件（selected_paths，必须为候选中的 relative_path）
2. 生成子 Agent 执行计划（检测参数、标签策略、是否视觉映射、annotation_scope 等）

子 Agent 工具链：run_object_detection → map_detection_boxes_to_labels → finalize_image_change

硬性要求：
- 只提交 map 成功且含 label_id 的框（allow_unlabeled_boxes=false）
- label_strategy=single_label_for_all_boxes 仅当用户明确要求所有框同一标签
- 标签名为具体实例（球员名等）且与 YOLO 类名不一致时，use_vision_mapping=true（需视觉探针通过）
- 用户说置信度/IoU/模型名时写入 detection_hints
- 勿编造不在候选列表中的路径；无法确定时 selected_paths 为空并在 scope_reason 说明

选图规则（selected_paths 由你根据本轮用户请求决定，每轮重新分析）：
- 以【用户请求】为最高优先级；用户纠正或扩大范围时，覆盖对话历史中的旧范围
- 用户指定文件夹/目录（如「data 下所有图片」）时，从候选列表按 parent 字段选出该目录下全部图片
- 用户指定单张或若干具体文件时，只选对应 path；指代消解可参考【对话上下文】（如「第一张」「按上面说的」）
- 「当前打开文件」仅作参考；用户未限定为单张时，不要仅因当前打开图而只选一张
- scope_reason 必须与 selected_paths 的实际数量和内容一致

只输出一个 JSON 对象：
{
  "selected_paths": ["relative/path.jpg"],
  "scope_reason": "为何选这些图",
  "intent_summary": "一句话任务摘要",
  "label_strategy": "map_each_box_to_label|single_label_for_all_boxes",
  "use_vision_mapping": false,
  "detection_hints": {"needs_object_detection": true, "model_id": null, "conf_threshold": null, "iou_threshold": null, "notes": ""},
  "sub_agent_constraints": {"require_per_box_mapping": true, "allow_unlabeled_boxes": false, "min_labeled_box_count": 1},
  "annotation_scope": {"scope_summary": "", "include_detection_labels": [], "exclude_detection_labels": [], "include_label_names": [], "exclude_label_names": []},
  "plan_steps": ["简要步骤"]
}"""


def _filter_selected_paths(
    raw_paths: object,
    candidates: list[dict],
) -> tuple[list[str], str]:
    """将 LLM 输出的路径列表过滤为候选中存在的 relative_path。"""
    by_path: dict[str, dict] = {}
    for item in candidates:
        rel = str(item.get("relative_path") or "").strip().replace("\\", "/")
        if rel:
            by_path[rel] = item

    selected: list[str] = []
    if isinstance(raw_paths, list):
        for raw in raw_paths:
            key = str(raw).strip().replace("\\", "/")
            if key and key in by_path:
                selected.append(key)

    reason = ""
    return selected, reason


async def prepare_batch_annotation(
    llm: ChatOpenAI,
    *,
    user_request: str,
    current_relative_path: str,
    candidates: list[dict],
    label_candidates: list[dict],
    detection_models: list[dict],
    default_conf: float = 0.7,
    default_iou: float = 0.5,
    provider_is_vision: bool = False,
    project_name: str | None = None,
    label_names: list[str] | None = None,
    conversation_transcript: str = "",
    preselected_paths: list[str] | None = None,
) -> BatchPrepareResult:
    """单次 LLM 调用：选图 + 生成 BatchPlan，供后续逐张确定性 detect → map → finalize 使用。"""
    if not candidates:
        empty_scope = AnnotationScopePayload()
        plan = build_batch_plan_from_data(
            {},
            user_request=user_request,
            intent_summary=user_request[:200] or "批量图片标注",
            annotation_scope=empty_scope,
            label_candidates=label_candidates,
            detection_models=detection_models,
            default_conf=default_conf,
            default_iou=default_iou,
            provider_is_vision=provider_is_vision,
            log_prefix="batch-prepare",
        )
        return BatchPrepareResult(
            selected_paths=[],
            scope_reason="项目目录无图片候选",
            plan=plan,
        )

    prompt_candidates = [
        {
            "relative_path": c.get("relative_path"),
            "name": c.get("name"),
            "parent": c.get("parent"),
        }
        for c in candidates[:500]
    ]
    labels_line = ""
    if label_names:
        labels_line = f"\n项目标签：{', '.join(label_names[:40])}\n"

    # 仅当客户端显式传入 preselected_paths（如 UI 勾选）时锁定选图，不由回合理解注入
    locked_paths, _ = _filter_selected_paths(preselected_paths or [], candidates)
    locked_block = ""
    if locked_paths:
        locked_block = (
            f"\n【客户端已指定图片范围（勿修改 selected_paths）】\n"
            f"{json.dumps(locked_paths, ensure_ascii=False)}\n\n"
        )

    transcript_block = ""
    if conversation_transcript.strip():
        transcript_block = (
            f"\n【对话上下文】\n{conversation_transcript.strip()}\n\n"
        )

    messages = [
        SystemMessage(content=BATCH_PREPARE_SYSTEM),
        HumanMessage(
            content=(
                f"{transcript_block}"
                f"{locked_block}"
                f"用户请求：{user_request}\n\n"
                f"项目名称：{project_name or '未知'}\n"
                f"当前打开文件：{current_relative_path or '（无）'}"
                f"{labels_line}\n"
                f"候选图片（{len(prompt_candidates)} 项）：\n"
                f"{json.dumps(prompt_candidates, ensure_ascii=False)}\n\n"
                f"当前 LLM 视觉探针（API 实测）：{bool(provider_is_vision)}\n"
                f"标签是否需视觉映射：{labels_require_vision_mapping(label_candidates)}\n\n"
                f"检测模型：{json.dumps(detection_models[:10], ensure_ascii=False)}\n\n"
                f"标签候选：{json.dumps(label_candidates[:80], ensure_ascii=False)}"
            )
        ),
    ]

    resp = await llm.ainvoke(messages)
    content = resp.content if hasattr(resp, "content") else str(resp)
    data = extract_json_object(str(content))

    if locked_paths:
        selected = locked_paths
        scope_reason = str(data.get("scope_reason") or "").strip() or "客户端已指定图片范围"
    else:
        selected, _ = _filter_selected_paths(data.get("selected_paths"), candidates)
        scope_reason = str(data.get("scope_reason") or data.get("reason") or "").strip()
        if not selected and not scope_reason:
            scope_reason = "未从候选中解析到图片，请更具体说明文件夹或文件名"

    base_scope = merge_annotation_scope(
        AnnotationScopePayload(),
        user_request,
        label_names=label_names,
    )
    plan = build_batch_plan_from_data(
        data,
        user_request=user_request,
        intent_summary=str(data.get("intent_summary") or user_request[:200]),
        annotation_scope=base_scope,
        label_candidates=label_candidates,
        detection_models=detection_models,
        default_conf=default_conf,
        default_iou=default_iou,
        provider_is_vision=provider_is_vision,
        log_prefix="batch-prepare",
    )

    log_annotation_agent(
        "batch-prepare",
        "批量准备完成",
        selected_count=len(selected),
        paths=selected[:10],
        use_vision_mapping=plan.use_vision_mapping,
        label_strategy=plan.label_strategy,
    )

    return BatchPrepareResult(
        selected_paths=selected,
        scope_reason=scope_reason,
        plan=plan,
    )
