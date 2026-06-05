from app.agent.assist_vision import vision_relative_from_user_text


def test_vision_relative_from_user_text() -> None:
    assert vision_relative_from_user_text("data/1.jpg 是谁") == "data/1.jpg"
    assert vision_relative_from_user_text("看图") == ""
    assert vision_relative_from_user_text("path/to/photo.PNG 描述") == "path/to/photo.PNG"
