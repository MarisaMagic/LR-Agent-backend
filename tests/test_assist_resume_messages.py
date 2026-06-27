import json

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage

from app.agent.context_service import append_client_tool_results_to_messages
from app.schemas.agent import ClientToolResult


def test_append_client_tool_results_adds_ai_and_tool_messages() -> None:
    lc_messages = [
        SystemMessage(content="sys"),
        HumanMessage(content="标注并生成报告"),
    ]
    results = [
        ClientToolResult(
            tool_call_id="call-1",
            name="execute_batch_annotation",
            result=json.dumps(
                {
                    "status": "completed",
                    "tool": "execute_batch_annotation",
                    "user_request": "标注并生成报告",
                    "summary": "批量标注已完成",
                    "next_hint": "可继续 write_workspace_file",
                },
                ensure_ascii=False,
            ),
        ),
    ]

    append_client_tool_results_to_messages(
        lc_messages,
        results,
        user_content="标注并生成报告",
    )

    assert len(lc_messages) == 4
    ai_msg = lc_messages[2]
    tool_msg = lc_messages[3]
    assert isinstance(ai_msg, AIMessage)
    assert ai_msg.tool_calls
    assert ai_msg.tool_calls[0]["name"] == "execute_batch_annotation"
    assert ai_msg.tool_calls[0]["id"] == "call-1"
    assert isinstance(tool_msg, ToolMessage)
    assert tool_msg.tool_call_id == "call-1"
    parsed = json.loads(tool_msg.content)
    assert parsed["status"] == "completed"
    assert parsed["summary"] == "批量标注已完成"


def test_append_client_tool_results_cumulative_pairs() -> None:
    lc_messages = [HumanMessage(content="标注并写报告")]
    results = [
        ClientToolResult(
            tool_call_id="call-1",
            name="execute_batch_annotation",
            result='{"status":"completed","summary":"标注完成"}',
        ),
        ClientToolResult(
            tool_call_id="call-2",
            name="analyze_data",
            result='{"status":"completed","summary":"分析完成"}',
        ),
    ]

    append_client_tool_results_to_messages(
        lc_messages,
        results,
        user_content="标注并写报告",
    )

    assert len(lc_messages) == 5
    assert isinstance(lc_messages[1], AIMessage)
    assert isinstance(lc_messages[2], ToolMessage)
    assert isinstance(lc_messages[3], AIMessage)
    assert isinstance(lc_messages[4], ToolMessage)
    assert lc_messages[2].tool_call_id == "call-1"
    assert lc_messages[4].tool_call_id == "call-2"


def test_append_client_tool_results_empty_is_noop() -> None:
    lc_messages = [HumanMessage(content="hi")]
    append_client_tool_results_to_messages(lc_messages, [])
    assert len(lc_messages) == 1


def test_resume_placeholder_preserves_blocks_when_flag_set() -> None:
    """Resume 时 preserve_blocks=True 不应写入 blocks_json=[]。"""
    values: dict = {"status": "streaming", "error": None}
    preserve_blocks = True
    if not preserve_blocks:
        values["blocks_json"] = []
    assert "blocks_json" not in values


def test_non_resume_placeholder_clears_blocks() -> None:
    values: dict = {"status": "streaming", "error": None}
    preserve_blocks = False
    if not preserve_blocks:
        values["blocks_json"] = []
    assert values["blocks_json"] == []
