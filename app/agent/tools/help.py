"""LR-Agent 应用功能说明，供 get_lr_agent_help 工具返回。"""

LR_AGENT_HELP = """# LR-Agent 功能概览

- **资源管理器**：浏览本地文件夹与文件
- **标注任务**：创建与管理图像标注项目
- **预训练模型**：配置用于辅助标注的模型
- **大模型配置**：添加 OpenAI 兼容 API（如通义、DeepSeek）
- **Agent 面板**：右侧对话助手，支持工具查询账户、读取文件与当前界面上下文

**Ask 模式**：问答、查已有标注（read_file_annotation）、搜索代码（grep_workspace）、列目录（list_workspace_directory）、读文本/代码（read_workspace_file，支持行范围）、读文档（read_document_file）、看图（read_image_for_vision）；不写入标注文件。
**Agent 模式**：批量检测与提案；标注变更（改标签/删框）；数据分析（脚本预览后执行）；报告/文档（Markdown）。
执行批量写入或变更标注请描述明确图片范围；查某张图已有标注在 Ask 即可。"""


def get_lr_agent_help(topic: str | None = None) -> str:
    """按 topic 关键词返回功能说明；无 topic 时返回完整概览。"""
    if not topic or not topic.strip():
        return LR_AGENT_HELP
    needle = topic.strip().lower()
    if "标注" in needle or "annotation" in needle:
        return (
            "标注任务在左侧活动栏「标注任务」中创建；数据保存在 `.lr-agent/annotations/`。"
            " **Ask**：分析与建议，不执行批量标注。**Agent**：描述图片范围后执行检测与提案。"
        )
    if "模型" in needle or "model" in needle:
        return "预训练模型在左侧「预训练模型」面板配置，供标注工作区推理使用。"
    if "agent" in needle or "对话" in needle:
        return "Agent 在右侧活动栏打开，需登录并在「大模型配置」中添加 API Key。"
    return LR_AGENT_HELP
