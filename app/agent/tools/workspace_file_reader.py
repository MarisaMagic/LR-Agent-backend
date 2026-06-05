"""Read workspace text, documents, and images for agent tools."""

from __future__ import annotations

import json
from pathlib import Path

from PIL import Image

from app.agent.tools.workspace_path import resolve_workspace_file
from app.core.config import Settings
from app.schemas.agent import ClientContextInput

VISION_TOOL_NAME = "read_image_for_vision"
VISION_PATH_MARKER = "__vision_image_path__"

TEXT_BLOCKLIST_SUFFIXES = frozenset(
    {
        ".png",
        ".jpg",
        ".jpeg",
        ".gif",
        ".webp",
        ".bmp",
        ".ico",
        ".svg",
        ".pdf",
        ".docx",
        ".doc",
        ".zip",
        ".rar",
        ".7z",
        ".exe",
        ".dll",
        ".so",
        ".dylib",
        ".mp3",
        ".mp4",
        ".avi",
        ".mov",
        ".woff",
        ".woff2",
        ".ttf",
        ".otf",
    }
)

IMAGE_SUFFIXES = frozenset({".png", ".jpg", ".jpeg", ".gif", ".webp", ".bmp", ".ico"})
DOCUMENT_SUFFIXES = frozenset({".pdf", ".docx"})


def read_workspace_text_file(
    client_context: ClientContextInput | None,
    path: str,
    *,
    settings: Settings,
) -> str:
    resolved, err = resolve_workspace_file(client_context, path)
    if resolved is None:
        return err

    suffix = resolved.suffix.lower()
    if suffix in TEXT_BLOCKLIST_SUFFIXES:
        return (
            f"「{resolved.name}」不是纯文本文件。"
            f"图片请用 {VISION_TOOL_NAME}；PDF/DOCX 请用 read_document_file。"
        )

    max_bytes = settings.agent_read_file_max_bytes
    max_lines = settings.agent_read_file_max_lines

    try:
        size = resolved.stat().st_size
    except OSError as exc:
        return f"无法读取文件：{exc}"

    truncated = False
    try:
        if size > max_bytes:
            truncated = True
            raw = resolved.read_bytes()[:max_bytes]
        else:
            raw = resolved.read_bytes()
    except OSError as exc:
        return f"读取失败：{exc}"

    if b"\x00" in raw[:8192]:
        return f"「{resolved.name}」似乎是二进制文件，请使用对应专用工具。"

    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError:
        try:
            text = raw.decode("utf-8-sig")
        except UnicodeDecodeError:
            return f"「{resolved.name}」不是 UTF-8 文本，暂不支持读取。"

    lines = text.splitlines()
    if len(lines) > max_lines:
        truncated = True
        lines = lines[:max_lines]
        text = "\n".join(lines)

    rel_hint = resolved.name
    header = f"文件：{rel_hint}\n大小：{size} 字节\n"
    if truncated:
        header += f"（内容已截断，最多 {max_bytes} 字节 / {max_lines} 行）\n"
    header += "---\n"
    return header + text


def read_image_for_vision_tool(
    client_context: ClientContextInput | None,
    path: str,
    *,
    provider_is_vision: bool,
) -> str:
    if not provider_is_vision:
        return (
            "当前大模型未通过视觉能力探针，无法分析图片内容。"
            "请在「大模型配置」中选用支持视觉的模型并重探视觉能力。"
        )

    resolved, err = resolve_workspace_file(client_context, path)
    if resolved is None:
        return err

    suffix = resolved.suffix.lower()
    if suffix not in IMAGE_SUFFIXES:
        return f"「{resolved.name}」不是支持的图片格式（{', '.join(sorted(IMAGE_SUFFIXES))}）。"

    try:
        with Image.open(resolved) as img:
            width, height = img.size
            fmt = (img.format or suffix.lstrip(".")).upper()
    except OSError as exc:
        return f"无法打开图片：{exc}"

    payload = {
        "ok": True,
        "path": str(resolved),
        "name": resolved.name,
        "width": width,
        "height": height,
        "format": fmt,
        VISION_PATH_MARKER: str(resolved),
        "message": (
            f"已加载图片 {resolved.name}（{width}x{height} {fmt}）。"
            "系统将在本条工具结果后注入附图，请根据图像回答用户问题。"
        ),
    }
    return json.dumps(payload, ensure_ascii=False)


def read_document_file(
    client_context: ClientContextInput | None,
    path: str,
    *,
    settings: Settings,
) -> str:
    resolved, err = resolve_workspace_file(client_context, path)
    if resolved is None:
        return err

    suffix = resolved.suffix.lower()
    if suffix not in DOCUMENT_SUFFIXES:
        return f"「{resolved.name}」不是支持的文档格式（pdf、docx）。"

    max_pages = settings.agent_read_document_max_pages

    try:
        if suffix == ".pdf":
            text, meta = _extract_pdf_text(resolved, max_pages=max_pages)
        else:
            text, meta = _extract_docx_text(resolved)
    except Exception as exc:
        return f"解析文档失败：{exc}"

    if not text.strip():
        return f"「{resolved.name}」未能提取到文本（{meta}）。"

    max_chars = settings.agent_read_file_max_bytes
    truncated = len(text) > max_chars
    if truncated:
        text = text[:max_chars]

    header = f"文件：{resolved.name}\n{meta}\n"
    if truncated:
        header += f"（正文已截断至 {max_chars} 字符）\n"
    header += "---\n"
    return header + text


def _extract_pdf_text(path: Path, *, max_pages: int) -> tuple[str, str]:
    from pypdf import PdfReader

    reader = PdfReader(str(path))
    total = len(reader.pages)
    limit = min(total, max_pages)
    parts: list[str] = []
    for idx in range(limit):
        page = reader.pages[idx]
        parts.append(page.extract_text() or "")
    meta = f"PDF 共 {total} 页，已提取前 {limit} 页"
    return "\n\n".join(parts), meta


def _extract_docx_text(path: Path) -> tuple[str, str]:
    from docx import Document

    doc = Document(str(path))
    paragraphs = [p.text for p in doc.paragraphs if p.text.strip()]
    return "\n\n".join(paragraphs), f"DOCX 段落数：{len(paragraphs)}"


def extract_vision_path_from_tool_result(tool_name: str, result_text: str) -> str | None:
    if tool_name != VISION_TOOL_NAME:
        return None
    try:
        data = json.loads(result_text)
    except json.JSONDecodeError:
        return None
    if not isinstance(data, dict) or not data.get("ok"):
        return None
    path = str(data.get(VISION_PATH_MARKER) or "").strip()
    if not path:
        return None
    candidate = Path(path)
    return str(candidate) if candidate.is_file() else None


def format_vision_tool_result_for_display(result_text: str) -> str:
    try:
        data = json.loads(result_text)
    except json.JSONDecodeError:
        return result_text
    if not isinstance(data, dict):
        return result_text
    display = dict(data)
    display.pop(VISION_PATH_MARKER, None)
    return json.dumps(display, ensure_ascii=False, indent=2)
