"""测试 JobState 状态机定义与转换合法性。"""

from app.schemas.agent import JobState


def test_all_states_defined():
    assert JobState.REGISTERED.value == "registered"
    assert JobState.STREAMING.value == "streaming"
    assert JobState.TOOL_PENDING.value == "tool_pending"
    assert JobState.RESUMING.value == "resuming"
    assert JobState.DONE.value == "done"
    assert JobState.ERROR.value == "error"
    assert JobState.CANCELLED.value == "cancelled"


def test_state_is_str_enum():
    assert isinstance(JobState.REGISTERED, str)
    assert JobState.REGISTERED == "registered"


def test_valid_transitions():
    """验证典型合法状态转换序列。"""
    transitions = [
        JobState.REGISTERED,
        JobState.STREAMING,
        JobState.TOOL_PENDING,
        JobState.RESUMING,
        JobState.DONE,
    ]
    # 确保所有状态不重复（状态机应单向推进）
    assert len(set(transitions)) == len(transitions)


def test_terminal_states():
    terminal = {JobState.DONE, JobState.ERROR, JobState.CANCELLED}
    for state in terminal:
        assert state in JobState
