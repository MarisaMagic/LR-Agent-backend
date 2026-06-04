"""Unit tests for annotation scope resolution helpers (no LLM)."""
from app.agent.annotation.schemas import AnnotationScopePayload


def test_annotation_scope_defaults():
    scope = AnnotationScopePayload()
    assert scope.scope_summary == ""
    assert scope.include_detection_labels == []
