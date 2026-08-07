"""工具调用解析：LLM API tool_calls 优先，其次正文伪代码（覆盖全部注册工具）。"""

from __future__ import annotations

import re
import uuid
from dataclasses import dataclass

from app.agent.tools.tool_registry_meta import LOCAL_CANONICAL_TOOL_NAMES, TOOL_RUNNERS

_CLIENT_ASYNC_PATTERN = re.compile(
    r"(auto_annotate|mutate_annotation|analyze_data)\s*\(",
    re.IGNORECASE,
)

_WRITE_WORKSPACE_PATTERN = re.compile(
    r"write_workspace_file\s*\(",
    re.IGNORECASE,
)

_USER_REQUEST_PATTERN = re.compile(
    r"""user_request\s*=\s*(['"])(.+?)\1""",
    re.DOTALL,
)


@dataclass(frozen=True)
class ResolvedToolCall:
    tool_call_id: str
    name: str
    arguments: dict
    source: str  # api | parsed


ResolvedClientToolCall = ResolvedToolCall


def _is_in_code_block(text: str, match_start: int) -> bool:
    """检查位置 match_start 是否位于 Markdown 代码围栏（```...```）内部。"""
    in_block = False
    pos = 0
    while pos < match_start:
        idx = text.find("```", pos)
        if idx == -1 or idx > match_start:
            break
        in_block = not in_block
        pos = idx + 3
    return in_block


def registered_tool_names() -> frozenset[str]:
    return LOCAL_CANONICAL_TOOL_NAMES


def tool_mentioned_in_text(text: str) -> str | None:
    if not text.strip():
        return None
    for name in sorted(TOOL_RUNNERS.keys(), key=len, reverse=True):
        for m in re.finditer(rf"{re.escape(name)}\s*\(", text, re.IGNORECASE):
            if not _is_in_code_block(text, m.start()):
                return name
    return None


def client_tool_mentioned_in_text(text: str) -> str | None:
    from app.agent.tools.tool_registry_meta import ASYNC_TOOL_NAMES

    for name in sorted(ASYNC_TOOL_NAMES, key=len, reverse=True):
        if re.search(rf"{re.escape(name)}\s*\(", text, re.IGNORECASE):
            return name
    return None


def _build_async_tool_arguments(
    name: str,
    user_content: str,
    parsed_args: dict | None = None,
) -> dict:
    args: dict = {"user_request": user_content.strip()}
    if parsed_args:
        if isinstance(parsed_args.get("user_request"), str) and parsed_args["user_request"].strip():
            parsed_ur = parsed_args["user_request"].strip()
            if len(parsed_ur) >= len(user_content.strip()) * 0.5:
                args["user_request"] = parsed_ur
        scope = parsed_args.get("scope_hint")
        if isinstance(scope, str) and scope.strip():
            args["scope_hint"] = scope.strip()
    return args


def _parse_write_workspace_file_pseudo(text: str) -> dict[str, str] | None:
    match = _WRITE_WORKSPACE_PATTERN.search(text)
    if not match:
        return None
    rest = text[match.end() :]

    relative_path: str | None = None
    for pattern in (
        r'relative_path\s*=\s*"([^"]*)"',
        r"relative_path\s*=\s*'([^']*)'",
    ):
        rel_match = re.search(pattern, rest)
        if rel_match:
            relative_path = rel_match.group(1).strip()
            break
    if not relative_path:
        return None

    content: str | None = None
    for pattern in (
        r'content\s*=\s*"""(.*?)"""',
        r"content\s*=\s*'''(.*?)'''",
        r'content\s*=\s*"((?:[^"\\]|\\.)*)"',
        r"content\s*=\s*'((?:[^'\\]|\\.)*)'",
    ):
        content_match = re.search(pattern, rest, re.DOTALL)
        if content_match:
            content = content_match.group(1)
            break
    if content is None:
        return None

    return {"relative_path": relative_path, "content": content}


def parse_pseudo_tool_calls(
    text: str,
    user_content: str,
    *,
    completed_tools: frozenset[str] | None = None,
) -> list[ResolvedToolCall]:
    if not text.strip():
        return []

    completed = completed_tools or frozenset()
    calls: list[ResolvedToolCall] = []
    seen: set[str] = set()

    write_match = _WRITE_WORKSPACE_PATTERN.search(text)
    if write_match and not _is_in_code_block(text, write_match.start()):
        write_args = _parse_write_workspace_file_pseudo(text)
        if write_args and "write_workspace_file" not in completed:
            calls.append(
                ResolvedToolCall(
                    tool_call_id=f"lr-parsed-write-{uuid.uuid4().hex[:10]}",
                    name="write_workspace_file",
                    arguments=write_args,
                    source="parsed",
                )
            )
            seen.add("write_workspace_file")

    for match in _CLIENT_ASYNC_PATTERN.finditer(text):
        raw_name = match.group(1)
        if raw_name in seen or raw_name in completed:
            continue
        if _is_in_code_block(text, match.start()):
            continue
        seen.add(raw_name)

        snippet = text[match.start() : match.start() + 800]
        parsed_args: dict = {}
        ur_match = _USER_REQUEST_PATTERN.search(snippet)
        if ur_match:
            parsed_args["user_request"] = ur_match.group(2)

        calls.append(
            ResolvedToolCall(
                tool_call_id=f"lr-parsed-{uuid.uuid4().hex[:10]}",
                name=raw_name,
                arguments=_build_async_tool_arguments(raw_name, user_content, parsed_args),
                source="parsed",
            )
        )

    return calls


def normalize_api_tool_calls(
    api_tool_calls: list,
    *,
    completed_tools: frozenset[str] | None = None,
) -> list[ResolvedToolCall]:
    completed = completed_tools or frozenset()
    calls: list[ResolvedToolCall] = []
    for call in api_tool_calls or []:
        name = str(call.get("name") or "").strip()
        if not name or name in completed:
            continue
        tool_id = str(call.get("id") or f"tool-{uuid.uuid4().hex[:12]}")
        args = call.get("args") or {}
        if not isinstance(args, dict):
            try:
                import json

                args = json.loads(args) if args else {}
            except Exception:
                args = {}
        calls.append(
            ResolvedToolCall(
                tool_call_id=tool_id,
                name=name,
                arguments=args if isinstance(args, dict) else {},
                source="api",
            )
        )
    return calls


def resolve_tool_calls(
    *,
    api_tool_calls: list,
    response_text: str,
    user_content: str,
    completed_tools: frozenset[str] | None = None,
    enable_pseudo_parsing: bool = True,
) -> list[ResolvedToolCall]:
    completed = completed_tools or frozenset()

    api_resolved = normalize_api_tool_calls(
        api_tool_calls,
        completed_tools=completed,
    )
    if api_resolved:
        return api_resolved

    if not enable_pseudo_parsing:
        return []

    return parse_pseudo_tool_calls(
        response_text,
        user_content,
        completed_tools=completed,
    )


resolve_client_tool_calls = resolve_tool_calls
