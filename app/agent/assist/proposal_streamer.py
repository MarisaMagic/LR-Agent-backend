"""write_workspace_file tool_call_chunks 流式拦截器。

在 LLM astream 过程中，按 tc_index 递增解析 JSON 参数，
实时发射 file_proposal_start / file_proposal_delta SSE 事件。
"""

from collections.abc import AsyncIterator

from app.schemas.agent import StreamEventPayload

WRITE_TOOL_NAME = "write_workspace_file"


def _extract_json_string(buf: str, key: str) -> tuple[str | None, bool]:
    """从可能不完整的 JSON 缓冲区中提取字符串字段值。

    处理标准 JSON 转义（\\n \\t \\r \\\\ \\"），返回 (解码后的字符串, 是否已闭合)。
    若 key 尚未出现则返回 (None, False)；
    若出现但字符串值未闭合引号则返回 (已累积部分, False)；
    若字符串值已闭合引号则返回 (完整值, True)。
    """
    idx = buf.find(f'"{key}"')
    if idx == -1:
        return None, False
    try:
        colon = buf.index(":", idx)
        quote = buf.index('"', colon + 1)
    except ValueError:
        return None, False
    result: list[str] = []
    i = quote + 1
    while i < len(buf):
        c = buf[i]
        if c == "\\" and i + 1 < len(buf):
            nxt = buf[i + 1]
            esc: dict[str, str] = {"n": "\n", "t": "\t", "r": "\r", "\\": "\\", '"': '"'}
            result.append(esc.get(nxt, nxt))
            i += 2
        elif c == '"':
            return "".join(result), True
        else:
            result.append(c)
            i += 1
    return ("".join(result) if result else None), False


class ProposalStreamInterceptor:
    """拦截 LLM astream 的 tool_call_chunks，实时生成 file_proposal SSE。"""

    def __init__(self) -> None:
        # fp_states: tc_index → {args_buf, title_sent, content_sent_len, rel_path}
        self.fp_states: dict[int, dict] = {}

    def on_chunk(self, chunk) -> list[StreamEventPayload]:
        """处理一个 AIMessageChunk，返回需要产出的 SSE 事件。"""
        events: list[StreamEventPayload] = []
        for tc in chunk.tool_call_chunks or []:
            tc_name = tc.get("name")
            tc_args = tc.get("args") or ""
            tc_idx = tc.get("index")

            if tc_idx is None:
                continue

            if tc_name == WRITE_TOOL_NAME:
                self.fp_states.setdefault(tc_idx, {
                    "args_buf": "",
                    "title_sent": False,
                    "content_sent_len": 0,
                    "rel_path": "",
                })
                self.fp_states[tc_idx]["args_buf"] = tc_args
            elif tc_idx in self.fp_states and tc_args:
                state = self.fp_states[tc_idx]
                if tc_args.startswith(state["args_buf"]):
                    state["args_buf"] = tc_args
                else:
                    state["args_buf"] += tc_args

            if tc_idx in self.fp_states and self.fp_states[tc_idx]["args_buf"]:
                state = self.fp_states[tc_idx]
                if not state["title_sent"]:
                    rel_path, rel_closed = _extract_json_string(
                        state["args_buf"], "relative_path"
                    )
                    if rel_path and rel_closed:
                        state["title_sent"] = True
                        state["rel_path"] = rel_path
                        events.append(
                            StreamEventPayload(
                                type="file_proposal_start",
                                summary=rel_path,
                                image_path=rel_path,
                                detail="0",
                            )
                        )

                content, _ = _extract_json_string(state["args_buf"], "content")
                if content is not None and len(content) > state["content_sent_len"]:
                    delta = content[state["content_sent_len"]:]
                    state["content_sent_len"] = len(content)
                    if delta:
                        events.append(
                            StreamEventPayload(
                                type="file_proposal_delta",
                                content=delta,
                                image_path=state.get("rel_path"),
                            )
                        )
        return events

    def collected_paths(self) -> set[str]:
        """返回拦截器已发送过 file_proposal_start 的 relative_path 集合。"""
        streamed: set[str] = set()
        for state in self.fp_states.values():
            args_buf = state.get("args_buf", "")
            rel_path, rel_closed = _extract_json_string(args_buf, "relative_path")
            if rel_path and rel_closed and state.get("title_sent"):
                streamed.add(rel_path)
        return streamed
