from app.core.config import get_settings
from app.services.llm_secrets import KEY_ID_ACTIVE, KEY_ID_LEGACY, decrypt_api_key, encrypt_api_key, is_masked_api_key


def test_encrypt_decrypt_v1_roundtrip() -> None:
    settings = get_settings()
    plain = "sk-test-roundtrip-key-12345"
    cipher = encrypt_api_key(settings, plain, key_id=KEY_ID_ACTIVE)
    assert decrypt_api_key(settings, cipher, KEY_ID_ACTIVE) == plain


def test_legacy_v0_roundtrip() -> None:
    settings = get_settings()
    plain = "sk-legacy-key-abcdef"
    cipher = encrypt_api_key(settings, plain, key_id=KEY_ID_LEGACY)
    assert decrypt_api_key(settings, cipher, KEY_ID_LEGACY) == plain


def test_is_masked_api_key() -> None:
    assert is_masked_api_key("sk-****1234")
    assert is_masked_api_key("")
    assert not is_masked_api_key("sk-live-real-key-12345678")
