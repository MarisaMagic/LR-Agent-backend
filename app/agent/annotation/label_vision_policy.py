"""标签视觉映射策略：判断任务标签是否必须启用视觉映射。

当项目标签名与 YOLO/COCO 标准检测类名无交集时（如球员姓名 Curry/James vs person），
启发式类名匹配无法工作，batch_prepare / plan_parsing 会据此建议 use_vision_mapping=true。
"""

from __future__ import annotations

# COCO / YOLO 常见检测类名（小写），用于与项目标签名比对
YOLO_COCO_CLASS_NAMES = frozenset(
    {
        "person",
        "bicycle",
        "car",
        "motorcycle",
        "airplane",
        "bus",
        "train",
        "truck",
        "boat",
        "traffic light",
        "fire hydrant",
        "stop sign",
        "parking meter",
        "bench",
        "bird",
        "cat",
        "dog",
        "horse",
        "sheep",
        "cow",
        "elephant",
        "bear",
        "zebra",
        "giraffe",
        "backpack",
        "umbrella",
        "handbag",
        "tie",
        "suitcase",
        "frisbee",
        "skis",
        "snowboard",
        "sports ball",
        "kite",
        "baseball bat",
        "baseball glove",
        "skateboard",
        "surfboard",
        "tennis racket",
        "bottle",
        "wine glass",
        "cup",
        "fork",
        "knife",
        "spoon",
        "bowl",
        "banana",
        "apple",
        "sandwich",
        "orange",
        "broccoli",
        "carrot",
        "hot dog",
        "pizza",
        "donut",
        "cake",
        "chair",
        "couch",
        "potted plant",
        "bed",
        "dining table",
        "toilet",
        "tv",
        "laptop",
        "mouse",
        "remote",
        "keyboard",
        "cell phone",
        "microwave",
        "oven",
        "toaster",
        "sink",
        "refrigerator",
        "book",
        "clock",
        "vase",
        "scissors",
        "teddy bear",
        "hair drier",
        "toothbrush",
        "face",
    }
)


def _norm_label(name: str) -> str:
    return (name or "").strip().lower().replace("_", " ")


def labels_require_vision_mapping(label_candidates: list[dict]) -> bool:
    """判断标签候选是否需视觉映射：无标签名命中标准检测类名时返回 True。"""
    names = [_norm_label(str(c.get("name") or "")) for c in label_candidates]
    names = [n for n in names if n]
    if not names:
        return False
    return not any(n in YOLO_COCO_CLASS_NAMES for n in names)
