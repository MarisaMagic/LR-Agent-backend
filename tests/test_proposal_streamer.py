"""测试 ProposalStreamInterceptor 的 JSON 增量解析。"""

from app.agent.assist.proposal_streamer import (
    ProposalStreamInterceptor,
    _extract_json_string,
)


class TestExtractJsonString:
    def test_complete_string(self):
        buf = '{"relative_path": "reports/summary.md", "content": "hello"}'
        val, closed = _extract_json_string(buf, "relative_path")
        assert val == "reports/summary.md"
        assert closed is True

    def test_incomplete_string(self):
        buf = '{"relative_path": "reports/summary'
        val, closed = _extract_json_string(buf, "relative_path")
        assert val == "reports/summary"
        assert closed is False

    def test_key_not_present(self):
        buf = '{"other": "value"}'
        val, closed = _extract_json_string(buf, "relative_path")
        assert val is None
        assert closed is False

    def test_escaped_quotes(self):
        buf = '{"relative_path": "path/with\\"quote.jpg"}'
        val, closed = _extract_json_string(buf, "relative_path")
        assert val == 'path/with"quote.jpg'
        assert closed is True

    def test_content_extraction(self):
        buf = '{"content": "hello world"}'
        val, closed = _extract_json_string(buf, "content")
        assert val == "hello world"
        assert closed is True


class TestProposalStreamInterceptor:
    def _make_chunk(self, index, name, args):
        """构造一个 mock AIMessageChunk。"""
        class MockChunk:
            tool_call_chunks = []
        chunk = MockChunk()
        chunk.tool_call_chunks = [
            {"index": index, "name": name, "args": args, "id": None}
        ]
        return chunk

    def test_write_tool_starts_state(self):
        interceptor = ProposalStreamInterceptor()
        chunk = self._make_chunk(0, "write_workspace_file", '{"relative_path": "out.md"')
        events = interceptor.on_chunk(chunk)
        # 路径尚未闭合，不应有 file_proposal_start
        assert len(events) == 0
        assert 0 in interceptor.fp_states

    def test_path_closed_emits_start(self):
        interceptor = ProposalStreamInterceptor()
        chunk = self._make_chunk(0, "write_workspace_file", '{"relative_path": "out.md"}')
        events = interceptor.on_chunk(chunk)
        assert any(e.type == "file_proposal_start" for e in events)

    def test_non_write_tool_ignored(self):
        interceptor = ProposalStreamInterceptor()
        chunk = self._make_chunk(1, "read_workspace_file", '{"relative_path": "x.txt"}')
        events = interceptor.on_chunk(chunk)
        assert len(events) == 0
        assert 1 not in interceptor.fp_states

    def test_collected_paths(self):
        interceptor = ProposalStreamInterceptor()
        chunk = self._make_chunk(0, "write_workspace_file", '{"relative_path": "a.md"}')
        interceptor.on_chunk(chunk)
        paths = interceptor.collected_paths()
        assert "a.md" in paths
