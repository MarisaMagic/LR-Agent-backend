"""测试 VisionAutoLoader 决策逻辑。"""

from app.agent.assist.vision_bootstrap import VisionAutoLoader
from app.schemas.agent import ClientContextInput


def test_should_load_false_when_resume():
    loader = VisionAutoLoader(
        vision_fn=None,
        provider_is_vision=True,
        settings=None,
        client_context=None,
        user_content="",
    )
    assert loader.should_load(is_resume=True) is False


def test_should_load_false_when_no_vision_provider():
    loader = VisionAutoLoader(
        vision_fn=None,
        provider_is_vision=False,
        settings=None,
        client_context=None,
        user_content="data/1.jpg",
    )
    assert loader.should_load(is_resume=False) is False


def test_should_load_false_when_vision_fn_is_none():
    loader = VisionAutoLoader(
        vision_fn=None,
        provider_is_vision=True,
        settings=None,
        client_context=None,
        user_content="data/1.jpg",
    )
    assert loader.should_load(is_resume=False) is False


def test_should_load_true_with_image_path_in_user_content():
    loader = VisionAutoLoader(
        vision_fn=lambda x: x,  # mock
        provider_is_vision=True,
        settings=None,
        client_context=None,
        user_content="看看 data/1.jpg 这张图",
    )
    assert loader.should_load(is_resume=False) is True


def test_should_load_false_when_turn_understanding_says_no():
    class MockTU:
        needs_vision_input = False
    ctx = ClientContextInput(turn_understanding=MockTU())
    loader = VisionAutoLoader(
        vision_fn=lambda x: x,
        provider_is_vision=True,
        settings=None,
        client_context=ctx,
        user_content="data/1.jpg",
    )
    assert loader.should_load(is_resume=False) is False
