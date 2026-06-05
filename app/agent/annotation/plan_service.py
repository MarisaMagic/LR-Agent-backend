from __future__ import annotations

import json

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

from app.agent.annotation.json_utils import extract_json_object
from app.agent.annotation.label_vision_policy import labels_require_vision_mapping
from app.agent.annotation.plan_parsing import build_batch_plan_from_data
from app.agent.annotation.schemas import AnnotationScopePayload, BatchPlanResult

PLAN_SYSTEM = """你是批量图片矩形框标注规划助手。
任务类型为 bbox（轴对齐矩形框）。

职责：根据用户请求、意图 JSON、annotation_scope、标签列表与检测模型，生成子 Agent 执行计划。
每张图由独立 ReAct 子 Agent 调用工具：run_object_detection → map_detection_boxes_to_labels → finalize_image_change。

硬性要求：
- 最终批量 proposal 必须可直接应用；禁止「未分配标签、人工后续关联」
- 模式 A（allow_unlabeled_boxes=false）：只提交 map 成功且含 label_id 的框；未映射检测框丢弃
- 成功映射数 ≥ min_labeled_box_count 即可提交本图，不要求每个检测框都映射
- label_strategy=single_label_for_all_boxes 仅当用户明确要求所有框同一具体标签
- 用户目标是通用检测类但标签树只有具体实例标签时，必须用 map_each_box_to_label
- 当 label_strategy=map_each_box_to_label 且标签名与 YOLO 检测类名不一致时，use_vision_mapping=true（需视觉探针通过的模型）
- detection_hints：用户说置信度 0.x → conf_threshold；IoU 0.x → iou_threshold；指定模型 → model_id
- annotation_scope 与意图一致；勿编造用户未提及的检测类

输出 JSON：
{
  "intent_summary": "...",
  "label_strategy": "map_each_box_to_label|single_label_for_all_boxes",
  "use_vision_mapping": false,
  "detection_hints": {"needs_object_detection": true, "model_id": null, "conf_threshold": null, "iou_threshold": null, "notes": ""},
  "sub_agent_constraints": {"require_per_box_mapping": true, "allow_unlabeled_boxes": false, "min_labeled_box_count": 1},
  "annotation_scope": {"scope_summary": "", "include_detection_labels": [], "exclude_detection_labels": [], "include_label_names": [], "exclude_label_names": []},
  "plan_steps": ["步骤1", "步骤2"]
}"""


async def create_batch_plan(
    llm: ChatOpenAI,
    *,
    user_request: str,
    intent_summary: str,
    annotation_scope: AnnotationScopePayload,
    label_candidates: list[dict],
    detection_models: list[dict],
    image_count: int,
    default_conf: float = 0.7,
    default_iou: float = 0.5,
    provider_is_vision: bool = False,
) -> BatchPlanResult:
    messages = [
        SystemMessage(content=PLAN_SYSTEM),
        HumanMessage(
            content=(
                f"用户请求：{user_request}\n\n"
                f"意图摘要：{intent_summary}\n\n"
                f"annotation_scope：{json.dumps(annotation_scope.model_dump(), ensure_ascii=False)}\n\n"
                f"目标图片数：{image_count}\n\n"
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
    return build_batch_plan_from_data(
        data,
        user_request=user_request,
        intent_summary=intent_summary,
        annotation_scope=annotation_scope,
        label_candidates=label_candidates,
        detection_models=detection_models,
        default_conf=default_conf,
        default_iou=default_iou,
        provider_is_vision=provider_is_vision,
        log_prefix="create-plan",
    )
