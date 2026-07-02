"""客户端上下文提取与格式化工具。

从 ClientContextInput 中统一读取标注项目、工作区路径等字段，供下游模块复用：
  - tools/workspace_path、tools/registry：解析项目目录作为文件读取的根路径
"""

from __future__ import annotations

from app.schemas.agent import ClientContextInput


def normalize_rel_path(path: str) -> str:
    """将相对路径统一为正斜杠格式，去除空段（如 a\\b → a/b）。"""
    return "/".join(p for p in path.replace("\\", "/").split("/") if p)


def label_names_from_context(client_context: ClientContextInput | None) -> list[str]:
    """从标注项目快照提取标签名称列表，最多 40 个。"""
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
    """将标注项目快照格式化为多行文本，供回合理解 LLM 读取界面状态。"""
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
    """获取客户端当前打开文件的相对路径（已规范化），无则返回 None。"""
    if client_context is None:
        return None
    rel = (client_context.active_relative_path or "").strip()
    if rel:
        return normalize_rel_path(rel)
    return None


def project_directory(client_context: ClientContextInput | None) -> str | None:
    """获取标注项目目录绝对路径，优先 client_context，其次快照内字段。"""
    if client_context is None:
        return None
    direct = (client_context.project_directory_path or "").strip()
    if direct:
        return direct
    snap = client_context.annotation_project_snapshot
    if snap and getattr(snap, "project_directory_path", None):
        return str(snap.project_directory_path).strip() or None
    return None
