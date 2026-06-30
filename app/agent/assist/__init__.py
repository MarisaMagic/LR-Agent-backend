"""Assist 模式流式推理子模块。

职责划分：
  - tool_loop        多轮 LLM ↔ 工具循环
  - vision_bootstrap 视觉预加载决策与执行
  - proposal_streamer write_workspace_file chunk 流式拦截与 file_proposal SSE 生成
  - pending_emitter   异步工具 pending SSE 发送
"""
