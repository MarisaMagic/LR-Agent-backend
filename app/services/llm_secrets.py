import base64
import hashlib

from cryptography.fernet import Fernet, InvalidToken

from app.core.config import Settings

KEY_ID_LEGACY = "v0"
KEY_ID_ACTIVE = "v1"


def _fernet_from_secret_key(secret_key: str) -> Fernet:
    digest = hashlib.sha256(secret_key.encode("utf-8")).digest()
    return Fernet(base64.urlsafe_b64encode(digest))


def _resolve_v1_fernet(settings: Settings) -> Fernet:
    raw = settings.llm_secrets_master_key
    if raw:
        return Fernet(raw.encode("utf-8"))
    if settings.app_env == "production":
        raise ValueError("llm_secrets_master_key_required")
    return _fernet_from_secret_key(settings.secret_key)


def is_masked_api_key(value: str) -> bool:
    stripped = value.strip()
    if not stripped:
        return True
    return "*" in stripped


def encrypt_api_key(settings: Settings, plaintext: str, *, key_id: str = KEY_ID_ACTIVE) -> str:
    if key_id == KEY_ID_LEGACY:
        fernet = _fernet_from_secret_key(settings.secret_key)
    else:
        fernet = _resolve_v1_fernet(settings)
    return fernet.encrypt(plaintext.encode("utf-8")).decode("utf-8")


def decrypt_api_key(settings: Settings, ciphertext: str, key_id: str | None) -> str:
    kid = key_id or KEY_ID_LEGACY
    if kid == KEY_ID_LEGACY:
        fernet = _fernet_from_secret_key(settings.secret_key)
    else:
        fernet = _resolve_v1_fernet(settings)
    try:
        return fernet.decrypt(ciphertext.encode("utf-8")).decode("utf-8")
    except InvalidToken as exc:
        raise ValueError("invalid_encrypted_api_key") from exc
