from typing import Literal

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

from app.agent.annotation.llm_invoke import invoke_json_model

ROUTER_SYSTEM = """你是意图分类器。根据用户最新输入判断应答模式。

输出 JSON：
- mode：chat 或 assist（assist=需要账户/项目上下文/工具）
- domain：general | annotation | models | files | account
- confidence：0~1
- reason：简短说明

批量标注由客户端 Annotation 模式处理，不要将 mode 设为 annotate_batch。"""


class RouteDecision(BaseModel):
    mode: Literal["chat", "assist"] = "chat"
    confidence: float = Field(ge=0.0, le=1.0, default=0.8)
    domain: Literal["general", "annotation", "models", "files", "account"] = "general"
    reason: str = ""


async def classify_intent(llm: ChatOpenAI, user_content: str) -> RouteDecision:
    decision = await invoke_json_model(
        llm,
        [
            SystemMessage(content=ROUTER_SYSTEM),
            HumanMessage(content=user_content),
        ],
        RouteDecision,
    )
    if decision.confidence < 0.6:
        decision.mode = "chat"
    return decision
