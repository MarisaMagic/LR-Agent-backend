"""工具守门员：程序化约束替代 prompt 中的行为规则。

在 LLM 输出文本中检测不合规声明（如未调用 write_workspace_file 却声称已保存），
标记 warning 供上层日志或降级处理。
"""

import re

# 匹配文件已写入声明（但未实际调用 write_workspace_file 时触发）
FILE_SAVED_PATTERNS: list[re.Pattern] = [
    re.compile(r"(已保存|已写入|已创建|已生成|文件已在)\s", re.IGNORECASE),
    re.compile(r"(文件位于|保存在了|写入到了)\s", re.IGNORECASE),
    re.compile(r"(file saved to|written to|created at)\s", re.IGNORECASE),
]

# 匹配伪代码调用模式（正文中写了 tool_name(...) 但未发起 API tool_call）
PSEUDO_CODE_PATTERN = re.compile(
    r"(execute_batch_annotation|mutate_annotation|analyze_data)\s*\(",
    re.IGNORECASE,
)

# 匹配无视觉探针时声称按像素分析图片
NO_VISION_CLAIM_PATTERN = re.compile(
    r"(根据图片|从图片来看|图中显示|图片展示|画面中|像素上看)",
    re.IGNORECASE,
)


class ToolGuard:
    """程序化规则替代 prompt 中的大段约束文本。"""

    @staticmethod
    def audit_output(
        text: str,
        completed_tools: frozenset[str],
        provider_is_vision: bool = True,
    ) -> list[str]:
        """对 LLM 输出文本进行合规审计，返回 warning 列表。"""
        warnings: list[str] = []
        if not text.strip():
            return warnings

        write_done = "write_workspace_file" in completed_tools
        if not write_done:
            for pattern in FILE_SAVED_PATTERNS:
                if pattern.search(text):
                    warnings.append(
                        "检测到文件写入声明，但 write_workspace_file 未实际调用。"
                    )
                    break

        if not provider_is_vision:
            if NO_VISION_CLAIM_PATTERN.search(text):
                warnings.append(
                    "检测到视觉分析声明，但当前模型未通过视觉探针。"
                )

        return warnings

    @staticmethod
    def detect_pseudo_code(text: str) -> bool:
        """检测文本中是否包含工具伪代码调用。"""
        return bool(PSEUDO_CODE_PATTERN.search(text))
