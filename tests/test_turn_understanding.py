from unittest.mock import AsyncMock, patch

import pytest

from app.agent.turn_understanding_service import (
    TurnUnderstandingLlmResult,
    understand_turn,
    _normalize_turn_kind,
)
from app.schemas.agent import ClientContextInput


def test_normalize_turn_kind_maps_wants_batch_to_execute():
    assert _normalize_turn_kind("wants_batch") == "execute_batch"
    assert _normalize_turn_kind("execute_batch") == "execute_batch"
    assert _normalize_turn_kind("mutate_annotation") == "mutate_annotation"
    assert _normalize_turn_kind("analyze_data") == "analyze_data"
    assert _normalize_turn_kind("unknown") == "converse"


@pytest.mark.asyncio
async def test_understand_turn_llm_only():
    mock_llm = AsyncMock()
    mock_result = TurnUnderstandingLlmResult(
        resolved_user_content="标注 data/1.jpg 与 data/2.jpg",
        referenced_relative_paths=["data/1.jpg", "data/2.jpg"],
        resolved_active_relative_path="data/1.jpg",
        task_intent="execute_batch",
        turn_kind="execute_batch",
        needs_vision_input=False,
        confidence=0.9,
        scope_notes="两张图",
        reason="llm",
    )

    with patch(
        "app.agent.turn_understanding_service.invoke_json_model",
        new_callable=AsyncMock,
        return_value=mock_result,
    ):
        result = await understand_turn(
            mock_llm,
            user_content="标注这两张图片",
            client_context=ClientContextInput(active_relative_path="data/1.jpg"),
            conversation_transcript="用户: 问 data 下 1.jpg 和 2.jpg\n助手: 好的",
            provider_is_vision=True,
        )

    assert result.turn_kind == "execute_batch"
    assert len(result.referenced_relative_paths) == 2
    assert result.reason == "llm"


@pytest.mark.asyncio
async def test_understand_turn_editor_mode_downgrades_annotation():
    mock_llm = AsyncMock()
    mock_result = TurnUnderstandingLlmResult(
        resolved_user_content="批量标注 data 目录",
        referenced_relative_paths=["data/1.jpg"],
        resolved_active_relative_path="data/1.jpg",
        task_intent="execute_batch",
        turn_kind="execute_batch",
        needs_vision_input=False,
        confidence=0.9,
        scope_notes="",
        reason="llm",
    )

    with patch(
        "app.agent.turn_understanding_service.invoke_json_model",
        new_callable=AsyncMock,
        return_value=mock_result,
    ):
        result = await understand_turn(
            mock_llm,
            user_content="批量标注",
            client_context=ClientContextInput(work_mode="editor"),
        )

    assert result.turn_kind == "converse"
    assert result.task_intent == "converse"


def test_turn_understanding_llm_result_coerces_null_scope_notes():
    parsed = TurnUnderstandingLlmResult.model_validate(
        {
            "resolved_user_content": "删除 2.jpg 中的标注",
            "turn_kind": "mutate_annotation",
            "task_intent": "mutate_annotation",
            "referenced_relative_paths": ["2.jpg"],
            "scope_notes": None,
            "reason": "用户指定图片",
        }
    )
    assert parsed.scope_notes == ""
    assert parsed.reason == "用户指定图片"
    assert parsed.turn_kind == "mutate_annotation"


def test_coerce_null_string_fields():
    from app.agent.annotation.llm_invoke import coerce_null_string_fields

    data = {"scope_notes": None, "reason": "ok", "turn_kind": "analyze_data"}
    fixed = coerce_null_string_fields(data, "scope_notes", "reason")
    assert fixed["scope_notes"] == ""
    assert fixed["reason"] == "ok"
