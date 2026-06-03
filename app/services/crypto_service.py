from app.services.llm_secrets import decrypt_api_key, encrypt_api_key, is_masked_api_key

__all__ = ["decrypt_api_key", "encrypt_api_key", "is_masked_api_key"]


def mask_api_key(api_key: str) -> str:
    if not api_key:
        return ""
    if len(api_key) <= 8:
        return "****"
    return f"{api_key[:3]}****{api_key[-4:]}"
