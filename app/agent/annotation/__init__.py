# ⚠️ 这些函数仅供前端流水线（batchOrchestrator / mutationOrchestrator）内部调用。
# 外部不应直接 import 或通过 API 暴露。
# 标注执行的唯一用户入口是 auto_annotate → agentJobRegistry。
from app.agent.annotation.annotation_scope import merge_annotation_scope
from app.agent.annotation.heuristic_map_service import heuristic_map_boxes
from app.agent.annotation.judge_labels_service import judge_detection_labels
from app.agent.annotation.map_labels_service import map_detection_boxes_to_labels_unified
from app.agent.annotation.batch_prepare_service import prepare_batch_annotation

__all__ = [
    "prepare_batch_annotation",
    "judge_detection_labels",
    "map_detection_boxes_to_labels_unified",
    "merge_annotation_scope",
    "heuristic_map_boxes",
]
