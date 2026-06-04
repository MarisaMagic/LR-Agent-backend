"""When task labels cannot match YOLO class names, vision mapping is required."""
from __future__ import annotations

# Common COCO / YOLO detection class names (lowercase)
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
    """
    True when no project label name is a standard detection class
    (e.g. Curry/James vs person) — heuristic mapping cannot work.
    """
    names = [_norm_label(str(c.get("name") or "")) for c in label_candidates]
    names = [n for n in names if n]
    if not names:
        return False
    return not any(n in YOLO_COCO_CLASS_NAMES for n in names)
