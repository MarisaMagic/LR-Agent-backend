"""集成测试：orchestrator → pending → resume mock。"""

# 需要完整的 FastAPI + DB + Redis 环境，使用默认标记跳过
import pytest

pytestmark = pytest.mark.integration


@pytest.mark.skip(reason="需要完整的 PostgreSQL + Redis 基础设施")
class TestResumeCycle:
    async def test_full_resume_cycle(self):
        """1. stream → tool_pending → 中断
           2. 前端执行 client_tool → 收集结果
           3. 新 POST 带 client_tool_results → resume
           4. LLM 收到 ToolMessage → 继续推理 → done
        """
        pass
