"""Tests for batch plan parsing defaults."""

from app.agent.annotation.plan_parsing import build_batch_plan_from_data
from app.agent.annotation.schemas import AnnotationScopePayload


def test_instance_labels_default_allow_unlabeled_boxes():
    label_candidates = [
        {"id": "a", "name": "斯蒂芬库里"},
        {"id": "b", "name": "勒布朗詹姆斯"},
    ]
    plan = build_batch_plan_from_data(
        {
            "use_vision_mapping": False,
            "sub_agent_constraints": {},
        },
        user_request="标注球星",
        intent_summary="标注球星",
        annotation_scope=AnnotationScopePayload(),
        label_candidates=label_candidates,
        detection_models=[{"id": "m1", "name": "YOLO"}],
        default_conf=0.7,
        default_iou=0.5,
        provider_is_vision=True,
    )
    assert plan.sub_agent_constraints.allow_unlabeled_boxes is True
    assert plan.use_vision_mapping is True


def test_coco_label_names_do_not_force_allow_unlabeled_by_default():
    label_candidates = [{"id": "p", "name": "person"}]
    plan = build_batch_plan_from_data(
        {"sub_agent_constraints": {}},
        user_request="标注人",
        intent_summary="标注人",
        annotation_scope=AnnotationScopePayload(),
        label_candidates=label_candidates,
        detection_models=[{"id": "m1"}],
        default_conf=0.7,
        default_iou=0.5,
        provider_is_vision=True,
    )
    assert plan.sub_agent_constraints.allow_unlabeled_boxes is False
