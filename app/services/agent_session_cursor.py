from datetime import datetime, timezone


def encode_session_cursor(updated_at_ms: int, session_id: str) -> str:
    return f"{updated_at_ms}:{session_id}"


def decode_session_cursor(cursor: str) -> tuple[datetime, str]:
    parts = cursor.split(":", 1)
    if len(parts) != 2 or not parts[0].isdigit():
        raise ValueError("invalid_cursor")
    updated_at_ms = int(parts[0])
    session_id = parts[1].strip()
    if not session_id:
        raise ValueError("invalid_cursor")
    updated_at = datetime.fromtimestamp(updated_at_ms / 1000, tz=timezone.utc)
    return updated_at, session_id
