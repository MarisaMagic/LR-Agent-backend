"""工作区文件读取与写入实现：文本/代码、文档（PDF/DOCX）、图片（视觉）、写文件提案。

由 registry 注册为 LLM 工具；视觉与写文件相关辅助函数供 assist_service / assist_vision 使用：
  - extract_vision_path_from_tool_result：从工具结果提取图片绝对路径
  - format_vision_tool_result_for_display：隐藏内部路径标记后展示给用户
  - extract_doc_proposal_from_tool_result：从写文件工具结果提取文档提案数据
"""

from __future__ import annotations

import json
from pathlib import Path

from PIL import Image

from app.agent.tools.tool_result import build_tool_result, format_tool_result_for_display
from app.agent.tools.workspace_path import resolve_workspace_file, resolve_workspace_write_path
from app.agent.tools.workspace_text_extensions import (
    TEXT_BLOCKLIST_SUFFIXES,
    is_allowed_text_extension,
)
from app.core.config import Settings
from app.schemas.agent import ClientContextInput

VISION_TOOL_NAME = "read_image_for_vision"
WRITE_TOOL_NAME = "write_workspace_file"
# 工具结果 JSON 中的内部字段，assist_service 据此注入多模态消息
VISION_PATH_MARKER = "__vision_image_path__"
# 工具结果 JSON 中的内部字段，assist_service 据此发出 file_proposal SSE 事件
DOC_PROPOSAL_MARKER = "__doc_proposal__"

IMAGE_SUFFIXES = frozenset({".png", ".jpg", ".jpeg", ".gif", ".webp", ".bmp", ".ico"})
DOCUMENT_SUFFIXES = frozenset({".pdf", ".docx"})


def read_workspace_text_file(
    client_context: ClientContextInput | None,
    path: str,
    *,
    settings: Settings,
    start_line: int | None = None,
    end_line: int | None = None,
) -> str:
    """读取 UTF-8 文本/代码文件，按配置截断字节数与行数；可选行范围（1-indexed，含首尾）。"""
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
    total_lines = len(lines)
    if len(lines) > max_lines:
        truncated = True
        lines = lines[:max_lines]

    line_range_applied = False
    if start_line is not None or end_line is not None:
        s = max(1, start_line if start_line is not None else 1)
        e = end_line if end_line is not None else len(lines)
        if e < s:
            return f"无效行范围：start_line ({s}) 不能大于 end_line ({e})。"
        lines = lines[s - 1 : e]
        line_range_applied = True
        range_label = f"L{s}-L{e}"
    else:
        range_label = ""

    text = "\n".join(lines)

    rel_hint = resolved.name
    header = f"文件：{rel_hint}\n大小：{size} 字节\n"
    if line_range_applied:
        header += f"行范围：{range_label}（共 {total_lines} 行）\n"
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
    """加载图片元信息并返回 JSON；assist_service 据此注入多模态用户消息。"""
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
    """提取 PDF / DOCX 正文，按配置截断页数与字符数。"""
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
    """从 read_image_for_vision 工具结果中提取图片绝对路径。"""
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
    """格式化视觉工具结果供前端展示，移除内部路径标记。"""
    try:
        data = json.loads(result_text)
    except json.JSONDecodeError:
        return result_text
    if not isinstance(data, dict):
        return result_text
    display = dict(data)
    display.pop(VISION_PATH_MARKER, None)
    return json.dumps(display, ensure_ascii=False, indent=2)


def write_workspace_file_tool(
    client_context: ClientContextInput | None,
    relative_path: str,
    content: str,
) -> str:
    """准备写文件提案：校验路径后返回 file_proposal 标记供 assist_service 转换为 SSE 事件。

    不直接写盘——写操作由前端在用户确认后执行。
    """
    resolved, err = resolve_workspace_write_path(client_context, relative_path)
    if resolved is None:
        return build_tool_result(
            ok=False,
            tool=WRITE_TOOL_NAME,
            status="error",
            summary=f"无法写入文件：{err}",
        )

    from app.agent.tools.workspace_path import allowed_roots
    rel_display = relative_path.strip()
    roots = allowed_roots(client_context)
    for root in roots:
        try:
            rel_display = str(resolved.relative_to(root)).replace("\\", "/")
            break
        except ValueError:
            continue

    suffix = resolved.suffix.lower()
    if not is_allowed_text_extension(suffix):
        return build_tool_result(
            ok=False,
            tool=WRITE_TOOL_NAME,
            status="error",
            summary=(
                f"write_workspace_file 不支持后缀 {suffix!r}（二进制/富媒体格式）。"
                f"请使用 UTF-8 文本或代码文件。"
            ),
        )

    title = resolved.stem.replace("-", " ").replace("_", " ").title()
    summary = (
        f"已生成文件提案：{rel_display}。"
        "文件尚未写入磁盘；用户确认（Keep All）后才会落盘。"
        "请勿在回复中声称文件已创建或已保存。"
    )
    return build_tool_result(
        ok=True,
        tool=WRITE_TOOL_NAME,
        status="proposal_ready",
        summary=summary,
        file_written=False,
        proposal_pending=True,
        **{
            DOC_PROPOSAL_MARKER: True,
            "relative_path": rel_display,
            "title": title,
            "content": content,
        },
    )


def extract_doc_proposal_from_tool_result(
    tool_name: str, result_text: str
) -> dict | None:
    """从 write_workspace_file 工具结果中提取文档提案数据。

    返回 {"relative_path": ..., "title": ..., "content": ...} 或 None。
    """
    if tool_name != WRITE_TOOL_NAME:
        return None
    try:
        data = json.loads(result_text)
    except json.JSONDecodeError:
        return None
    if not isinstance(data, dict) or not data.get(DOC_PROPOSAL_MARKER):
        return None
    return {
        "relative_path": str(data.get("relative_path") or "document.md"),
        "title": str(data.get("title") or "文档"),
        "content": str(data.get("content") or ""),
    }


def format_write_tool_result_for_display(result_text: str) -> str:
    return format_tool_result_for_display(result_text)
