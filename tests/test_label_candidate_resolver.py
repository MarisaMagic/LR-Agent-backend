"""Unit tests for label candidate resolver (no LLM)."""

import asyncio

from app.agent.annotation.annotation_scope import AnnotationScope
from app.agent.annotation.label_candidate_resolver import resolve_effective_label_candidates

ALL = [
    {"id": "a", "name": "curry"},
    {"id": "b", "name": "james"},
    {"id": "c", "name": "durant"},
    {"id": "d", "name": "jokic"},
    {"id": "e", "name": "giannis"},
]


def test_resolve_without_llm_keeps_full_pool():
    scope = AnnotationScope()

    async def _run():
        return await resolve_effective_label_candidates(
            None,
            all_candidates=ALL,
            scope=scope,
            user_request="hello",
            intent_summary="",
            box_count=3,
            image_bytes=None,
        )

    result = asyncio.run(_run())
    assert len(result.candidates) == len(ALL)
    assert result.source == "full"


def test_resolve_ignores_partial_names_in_intent_summary():
    """prepare 摘要举例（curry/james 等）不应缩小 map 候选池。"""
    scope = AnnotationScope()

    async def _run():
        return await resolve_effective_label_candidates(
            None,
            all_candidates=ALL,
            scope=scope,
            user_request="批量标注 data 文件夹",
            intent_summary="标注斯蒂芬库里、勒布朗詹姆斯等球员",
            box_count=2,
            image_bytes=None,
        )

    result = asyncio.run(_run())
    assert len(result.candidates) == len(ALL)
    assert result.source == "full"


def test_resolve_scope_include_label_names():
    scope = AnnotationScope(include_label_names=["curry", "durant"])

    async def _run():
        return await resolve_effective_label_candidates(
            None,
            all_candidates=ALL,
            scope=scope,
            user_request="mark curry and durant",
            intent_summary="",
            box_count=2,
            image_bytes=None,
        )

    result = asyncio.run(_run())
    names = {c["name"] for c in result.candidates}
    assert names == {"curry", "durant"}
    assert result.source == "scope"
