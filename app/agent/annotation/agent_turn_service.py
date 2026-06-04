"""Tool-bound LLM turns for client-executed annotation agents (fusion sub-agent tools)."""
from __future__ import annotations

import json
import uuid
from typing import Any, Literal

from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from langchain_core.tools import StructuredTool
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

AgentKind = Literal["scope", "image"]


class ToolCallOut(BaseModel):
    id: str
    name: str
    args: dict[str, Any] = Field(default_factory=dict)


class AgentTurnResult(BaseModel):
    content: str = ""
    tool_calls: list[ToolCallOut] = Field(default_factory=list)
    finish_reason: str | None = None


class MessageItem(BaseModel):
    role: Literal["system", "human", "assistant", "tool"]
    content: str = ""
    tool_call_id: str | None = None
    tool_calls: list[ToolCallOut] | None = None


SCOPE_SYSTEM = """你是批量图片标注的范围解析 Agent。

通过工具在项目目录内查找图片，不要猜测路径。

流程：
1. 用 glob_project_images 或 list_project_directory 查找符合用户描述的文件。
2. 确认后用 select_annotation_images 提交最终 relative_path 列表（可多张）。
3. 路径必须来自工具返回结果。"""

IMAGE_SYSTEM = """你是单张图片 bbox 标注子 Agent（fusion 工具链）。主流程已完成范围/计划解析。

工作流程（ReAct，每次只输出一步）：
1. 若需要重新检测，调用 run_object_detection；按 brief 写入 model_id/conf_threshold/iou_threshold。
2. 多目标 / require_per_box_mapping 时调用 map_detection_boxes_to_labels（use_vision_mapping=true 时会逐框视觉识别）。
3. map 成功后也可调用 finalize_image_change；服务端在 map 达标时会自动 finalize。
4. 成功映射数 ≥ min_labeled_box_count 方可完成；不足则 Final Answer 说明原因。

硬性规则：
- YOLO 检测类名 ≠ 任务标签名；不要标注 annotation_scope 外目标
- 禁止提交无 label_id 的框（allow_unlabeled_boxes=false）
- 不要编造坐标；使用 run_object_detection 返回的框
- 使用简体中文 Final Answer"""


def _scope_tool_schemas() -> list[StructuredTool]:
    class GlobInput(BaseModel):
        parent_folder: str = Field(default="", description="子目录如 data")
        name_pattern: str = Field(default="", description="文件名片段如 7.jpg")
        limit: int = Field(default=100, ge=1, le=500)

    class ListDirInput(BaseModel):
        relative_dir: str = Field(default="", description="相对子目录")
        max_entries: int = Field(default=80, ge=1, le=300)

    class SelectInput(BaseModel):
        selected_paths: list[str] = Field(default_factory=list)
        reason: str = ""

    def _stub(**_kwargs: Any) -> str:
        return "executed_on_client"

    return [
        StructuredTool.from_function(
            func=_stub,
            name="glob_project_images",
            description="在项目目录内 glob 图片，按文件夹与文件名过滤",
            args_schema=GlobInput,
        ),
        StructuredTool.from_function(
            func=_stub,
            name="list_project_directory",
            description="列出项目内某目录下的条目",
            args_schema=ListDirInput,
        ),
        StructuredTool.from_function(
            func=_stub,
            name="select_annotation_images",
            description="确认本轮要标注的图片相对路径列表",
            args_schema=SelectInput,
        ),
    ]


def _image_tool_schemas() -> list[StructuredTool]:
    class DetectInput(BaseModel):
        file_path: str = Field(default="", description="图片绝对路径")
        model_id: str = Field(default="", description="YOLO 模型 id")
        conf_threshold: float | None = Field(default=None, ge=0.05, le=0.95)
        iou_threshold: float | None = Field(default=None, ge=0.05, le=0.95)

    class MapInput(BaseModel):
        file_path: str = Field(default="", description="图片绝对路径")
        boxes: list[dict[str, Any]] = Field(
            default_factory=list,
            description="检测框列表（含 x,y,width,height,detection_label）",
        )

    class FinalizeInput(BaseModel):
        file_path: str = Field(default="", description="图片绝对路径")
        summary: str = Field(default="单文件标注")
        operation: str = Field(default="append")

    def _stub(**_kwargs: Any) -> str:
        return "executed_on_client"

    return [
        StructuredTool.from_function(
            func=_stub,
            name="run_object_detection",
            description="对单张图运行 YOLO 等 object_detection，返回像素或归一化检测框",
            args_schema=DetectInput,
        ),
        StructuredTool.from_function(
            func=_stub,
            name="map_detection_boxes_to_labels",
            description=(
                "为 run_object_detection 返回的每个框推荐 label_tree 中的 label_id；"
                "brief.use_vision_mapping=true 且为视觉模型时逐框裁剪识别"
            ),
            args_schema=MapInput,
        ),
        StructuredTool.from_function(
            func=_stub,
            name="finalize_image_change",
            description="提交最终标注；require_per_box_mapping 时可省略 boxes，由最近一次 map 生成",
            args_schema=FinalizeInput,
        ),
    ]


def tools_for_kind(kind: AgentKind) -> list[StructuredTool]:
    if kind == "scope":
        return _scope_tool_schemas()
    return _image_tool_schemas()


def system_for_kind(kind: AgentKind) -> str:
    return SCOPE_SYSTEM if kind == "scope" else IMAGE_SYSTEM


def messages_from_items(items: list[MessageItem]) -> list[BaseMessage]:
    out: list[BaseMessage] = []
    for item in items:
        if item.role == "system":
            out.append(SystemMessage(content=item.content))
        elif item.role == "human":
            out.append(HumanMessage(content=item.content))
        elif item.role == "assistant":
            tc = None
            if item.tool_calls:
                tc = [
                    {
                        "id": t.id,
                        "name": t.name,
                        "args": t.args,
                    }
                    for t in item.tool_calls
                ]
            out.append(AIMessage(content=item.content or "", tool_calls=tc or []))
        elif item.role == "tool":
            out.append(
                ToolMessage(
                    content=item.content,
                    tool_call_id=item.tool_call_id or "tool",
                )
            )
    return out


async def run_agent_turn(
    llm: ChatOpenAI,
    *,
    kind: AgentKind,
    messages: list[MessageItem],
) -> AgentTurnResult:
    tools = tools_for_kind(kind)
    llm_tools = llm.bind_tools(tools)
    lc_messages = messages_from_items(messages)
    if not any(isinstance(m, SystemMessage) for m in lc_messages):
        lc_messages = [SystemMessage(content=system_for_kind(kind)), *lc_messages]

    response = await llm_tools.ainvoke(lc_messages)
    tool_calls: list[ToolCallOut] = []
    for idx, call in enumerate(getattr(response, "tool_calls", None) or []):
        args = call.get("args") or {}
        if not isinstance(args, dict):
            try:
                args = json.loads(args) if args else {}
            except json.JSONDecodeError:
                args = {}
        call_id = str(call.get("id") or "").strip()
        if not call_id:
            call_id = f"{kind}-tool-{idx}-{uuid.uuid4().hex[:8]}"
        tool_calls.append(
            ToolCallOut(
                id=call_id,
                name=str(call.get("name") or ""),
                args=args,
            )
        )

    content = ""
    if hasattr(response, "content"):
        raw = response.content
        content = raw if isinstance(raw, str) else str(raw)

    return AgentTurnResult(content=content, tool_calls=tool_calls)
