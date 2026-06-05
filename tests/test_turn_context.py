from app.agent.turn_context import (
    build_turn_context_from_messages,
    format_turn_line,
    normalize_interaction_mode,
    turn_context_to_chat_inputs,
)
from app.schemas.agent import ChatMessageInput


def test_normalize_interaction_mode():
    assert normalize_interaction_mode("ask") == "chat"
    assert normalize_interaction_mode("chat") == "chat"
    assert normalize_interaction_mode("annotation") == "annotation"
    assert normalize_interaction_mode("annotate") == "annotation"


def test_format_turn_line_plain():
    assert format_turn_line(role="user", content="hi", interaction_mode="chat") == "用户: hi"
    assert format_turn_line(role="assistant", content="ok", interaction_mode="annotation") == "助手: ok"


def test_build_turn_context_window_and_transcript():
    messages = [
        ChatMessageInput(
            role="user",
            content="标注 data 第一张",
            message_id="m1",
            interaction_mode="chat",
        ),
        ChatMessageInput(
            role="assistant",
            content="好的",
            message_id="m2",
            interaction_mode="chat",
        ),
        ChatMessageInput(
            role="user",
            content="按上面标注",
            message_id="m3",
            interaction_mode="annotation",
        ),
    ]
    ctx = build_turn_context_from_messages(
        messages,
        current_user_content="按上面标注",
        max_turns_in_window=20,
        exclude_message_ids={"m4"},
    )
    assert len(ctx.windowed_lines) == 3
    assert "[Ask]" not in ctx.transcript
    assert "[Agent]" not in ctx.transcript
    assert "用户:" in ctx.transcript
    assert "第一张" in ctx.transcript


def test_turn_context_to_chat_inputs_no_prefix():
    messages = [
        ChatMessageInput(role="user", content="hello", message_id="u1", interaction_mode="chat"),
    ]
    ctx = build_turn_context_from_messages(messages, max_turns_in_window=20)
    inputs = turn_context_to_chat_inputs(ctx)
    assert inputs[0].content == "hello"
    assert "[Ask]" not in inputs[0].content
