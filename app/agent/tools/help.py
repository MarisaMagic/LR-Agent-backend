LR_AGENT_HELP = """# LR-Agent 功能概览

- **资源管理器**：浏览本地文件夹与文件
- **标注任务**：创建与管理图像标注项目
- **预训练模型**：配置用于辅助标注的模型
- **大模型配置**：添加 OpenAI 兼容 API（如通义、DeepSeek）
- **Agent 面板**：右侧对话助手，支持工具查询账户与当前界面上下文

在已打开的图片矩形框标注项目中，可在 Agent 对话中描述批量范围（如某文件夹、第 N 到 M 张），由 Agent 自动检测并生成候选标注，确认后应用。"""


def get_lr_agent_help(topic: str | None = None) -> str:
    if not topic or not topic.strip():
        return LR_AGENT_HELP
    needle = topic.strip().lower()
    if "标注" in needle or "annotation" in needle:
        return "标注任务在左侧活动栏「标注任务」中创建；项目数据保存在本机工作区 `.lr-agent/annotations/`。"
    if "模型" in needle or "model" in needle:
        return "预训练模型在左侧「预训练模型」面板配置，供标注工作区推理使用。"
    if "agent" in needle or "对话" in needle:
        return "Agent 在右侧活动栏打开，需登录并在「大模型配置」中添加 API Key。"
    return LR_AGENT_HELP
