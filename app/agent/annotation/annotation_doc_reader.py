"""Read per-file annotation JSON from project .lr-agent/annotations (Electron co-located API)."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

PROJECT_HIDDEN = ".lr-agent"
ANNOTATIONS_DIR = "annotations"
FILES_DIR = "files"


def normalize_relative_path(relative_path: str) -> str:
    return "/".join(part for part in relative_path.replace("\\", "/").split("/") if part)


def compute_file_key(relative_path: str) -> str:
    normalized = normalize_relative_path(relative_path)
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def annotations_files_dir(project_dir: str) -> Path:
    return Path(project_dir) / PROJECT_HIDDEN / ANNOTATIONS_DIR / FILES_DIR


def read_file_annotation_doc(project_dir: str, relative_path: str) -> tuple[dict | None, str]:
    """
    Return (document dict or None, error/detail message).
    """
    root = (project_dir or "").strip()
    if not root:
        return None, "未提供项目目录路径（project_directory_path）。"

    rel = normalize_relative_path(relative_path.strip())
    if not rel:
        return None, "请提供相对路径（如 data/2.jpg）。"

    base = Path(root)
    if not base.is_dir():
        return None, f"项目目录不可读：{root}"

    file_key = compute_file_key(rel)
    doc_path = annotations_files_dir(root) / f"{file_key}.json"
    if not doc_path.is_file():
        return None, f"未找到 {rel} 的标注文件（可能尚未标注）。"

    try:
        data = json.loads(doc_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return None, f"读取标注失败：{exc}"

    if not isinstance(data, dict):
        return None, "标注文件格式无效。"
    return data, ""
