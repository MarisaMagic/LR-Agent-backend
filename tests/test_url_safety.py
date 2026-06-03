import pytest

from app.core.config import get_settings
from app.services.url_safety import validate_llm_base_url


@pytest.mark.parametrize(
    "url",
    [
        "https://api.openai.com/v1",
        "https://dashscope.aliyuncs.com/compatible-mode/v1",
    ],
)
def test_validate_llm_base_url_allows_public_https(url: str) -> None:
    settings = get_settings()
    assert validate_llm_base_url(url, settings).startswith("https://")


@pytest.mark.parametrize(
    "url",
    [
        "http://127.0.0.1/v1",
        "https://localhost/v1",
        "http://169.254.169.254/latest",
    ],
)
def test_validate_llm_base_url_blocks_private(url: str) -> None:
    settings = get_settings()
    with pytest.raises(ValueError):
        validate_llm_base_url(url, settings)
