from __future__ import annotations

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

from app.agent.annotation.llm_invoke import invoke_json_model
from app.agent.annotation.schemas import AnnotationTaskParseResult

TASK_PARSE_SYSTEM = """你是 LR-Agent 的标注任务理解助手。

用户已通过 **Annotation 模式** 发起执行请求；你只需理解「如何标注」。

annotation_scope 描述「只标用户要的目标」（重要）：
- scope_summary：如「仅标注球员」「只标车辆」
- include_detection_labels：YOLO 类名白名单；用户明确只标某类检测目标时填写
- exclude_detection_labels：YOLO 类名黑名单；用户明确不要某类时填写
- include_label_names / exclude_label_names：任务标签名限制（可选）
- 未提及时 include/exclude 可留空；服务端会用用户原文做补充推断

输出 JSON 字段：
- intent_summary：一句话任务摘要
- needs_object_detection：bool，默认 true
- annotation_scope：上述对象"""


async def parse_annotation_task(
    llm: ChatOpenAI,
    *,
    user_request: str,
    project_name: str | None = None,
    annotation_type: str | None = None,
    label_names: list[str] | None = None,
) -> AnnotationTaskParseResult:
    labels_line = ""
    if label_names:
        labels_line = f"\n项目标签：{', '.join(label_names[:40])}\n"
    return await invoke_json_model(
        llm,
        [
            SystemMessage(content=TASK_PARSE_SYSTEM),
            HumanMessage(
                content=(
                    f"用户输入：{user_request}\n\n"
                    f"项目名称：{project_name or '未知'}\n"
                    f"标注类型：{annotation_type or '未知'}"
                    f"{labels_line}"
                )
            ),
        ],
        AnnotationTaskParseResult,
    )


async def classify_annotation_intent(
    llm: ChatOpenAI,
    *,
    user_request: str,
    has_active_project: bool,
    annotation_type: str | None,
    modality: str | None,
    interaction_mode: str = "annotation",
    label_names: list[str] | None = None,
    project_name: str | None = None,
) -> AnnotationTaskParseResult:
    del has_active_project, modality, interaction_mode
    parsed = await parse_annotation_task(
        llm,
        user_request=user_request,
        project_name=project_name,
        annotation_type=annotation_type,
        label_names=label_names,
    )
    if not parsed.intent_summary.strip():
        parsed.intent_summary = user_request.strip()[:200] or "批量图片标注"
    return parsed
