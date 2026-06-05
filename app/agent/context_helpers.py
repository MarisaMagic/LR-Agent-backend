"""Shared client / project context formatting for turn understanding and chat."""

from __future__ import annotations

from app.schemas.agent import ClientContextInput


def normalize_rel_path(path: str) -> str:
    return "/".join(p for p in path.replace("\\", "/").split("/") if p)


def label_names_from_context(client_context: ClientContextInput | None) -> list[str]:
    if client_context is None or client_context.annotation_project_snapshot is None:
        return []
    snap = client_context.annotation_project_snapshot
    names: list[str] = []
    for item in snap.labels or []:
        if isinstance(item, dict):
            name = str(item.get("name") or "").strip()
            if name:
                names.append(name)
    return names[:40]


def project_context_lines(client_context: ClientContextInput | None) -> str:
    if client_context is None or client_context.annotation_project_snapshot is None:
        return "当前无标注项目快照。"
    snap = client_context.annotation_project_snapshot
    labels = ", ".join(label_names_from_context(client_context)) or "（无）"
    active_rel = (client_context.active_relative_path or "").strip() or "（无）"
    return (
        f"项目：{snap.name or snap.project_id}\n"
        f"模态：{snap.modality}\n"
        f"标注类型：{snap.annotation_type}\n"
        f"标签：{labels}\n"
        f"当前打开文件（相对）：{active_rel}"
    )


def client_active_relative(client_context: ClientContextInput | None) -> str | None:
    if client_context is None:
        return None
    rel = (client_context.active_relative_path or "").strip()
    if rel:
        return normalize_rel_path(rel)
    return None


def project_directory(client_context: ClientContextInput | None) -> str | None:
    if client_context is None:
        return None
    direct = (client_context.project_directory_path or "").strip()
    if direct:
        return direct
    snap = client_context.annotation_project_snapshot
    if snap and getattr(snap, "project_directory_path", None):
        return str(snap.project_directory_path).strip() or None
    return None
