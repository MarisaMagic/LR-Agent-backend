"""context_snapshot 的 skills catalog 注入测试。"""

from app.agent.context_snapshot import (
    build_assist_system_prompt,
    format_skills_catalog_block,
)
from app.schemas.agent import ClientContextInput, SkillCatalogEntryInput


def _build_prompt(client_context: ClientContextInput | None) -> str:
    return build_assist_system_prompt(
        client_context,
        model="test-model",
        provider_label="测试",
        supports_vision=False,
    )


def test_injects_skills_catalog_block() -> None:
    ctx = ClientContextInput(
        workspace_root="/ws",
        skills_catalog=[
            SkillCatalogEntryInput(
                name="caveman",
                description="Ultra-compressed communication mode.",
            ),
            SkillCatalogEntryInput(
                name="docx",
                description="Create and edit Word documents.",
            ),
        ],
    )
    prompt = _build_prompt(ctx)

    assert "【可用 Skills】" in prompt
    assert "read_agent_skill" in prompt
    assert "- caveman: Ultra-compressed communication mode." in prompt
    assert "- docx: Create and edit Word documents." in prompt
    # 其他既有区块仍存在
    assert "【任务】" in prompt


def test_skips_empty_catalog() -> None:
    ctx = ClientContextInput(
        workspace_root="/ws",
        skills_catalog=[],
    )
    prompt = _build_prompt(ctx)

    assert "【可用 Skills】" not in prompt
    assert "read_agent_skill" not in prompt
    assert "【任务】" in prompt


def test_no_client_context_still_plain() -> None:
    prompt = _build_prompt(None)

    assert "【可用 Skills】" not in prompt


def test_format_block_skips_entries_missing_name_or_description() -> None:
    # 直接单测 format_skills_catalog_block 的防呆逻辑（绕过 pydantic 校验的畸形输入）
    block = format_skills_catalog_block(
        [
            SkillCatalogEntryInput(name="a", description="valid entry"),
            type(
                "BadEntry",
                (),
                {"name": "", "description": "no name"},
            )(),
            type(
                "BadEntry",
                (),
                {"name": "no-desc", "description": ""},
            )(),
        ]
    )

    assert "- a: valid entry" in block
    assert "no-desc" not in block
    assert "no name" not in block


def test_format_block_empty() -> None:
    assert format_skills_catalog_block([]) == ""
    assert format_skills_catalog_block(None) == ""
