from typing import Literal

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

ROUTER_SYSTEM = """你是意图分类器。根据用户最新输入，判断应答模式：
- chat：闲聊、概念解释、简单问答，不需要查账户或应用状态
- assist：需要查询账户、应用功能说明、或依赖用户当前工作区/打开文件/标注项目等上下文

domain 取 general / annotation / models / files / account 之一。
confidence 为 0~1。不确定时 mode=chat 且 confidence 偏低。"""


class RouteDecision(BaseModel):
    mode: Literal["chat", "assist"] = "chat"
    confidence: float = Field(ge=0.0, le=1.0, default=0.8)
    domain: Literal["general", "annotation", "models", "files", "account"] = "general"
    reason: str = ""


async def classify_intent(llm: ChatOpenAI, user_content: str) -> RouteDecision:
    structured = llm.with_structured_output(RouteDecision)
    result = await structured.ainvoke(
        [
            SystemMessage(content=ROUTER_SYSTEM),
            HumanMessage(content=user_content),
        ],
    )
    if isinstance(result, RouteDecision):
        decision = result
    else:
        decision = RouteDecision.model_validate(result)

    if decision.confidence < 0.6:
        decision.mode = "chat"
    return decision
