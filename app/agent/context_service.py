"""对话上下文服务：Assist resume 与 chat 系统提示词。"""

from langchain_core.messages import AIMessage, ToolMessage


CHAT_SYSTEM_PROMPT = """在 LR-Agent 系统内回答用户问题。结合【你的身份】中的模型信息作答，勿自称独立产品助手或其它未配置的模型。"""


def append_client_tool_results_to_messages(
    lc_messages: list,
    client_tool_results: list,
    *,
    user_content: str = "",
) -> list:
    """Resume 时在消息链末尾追加 AIMessage(tool_calls) + ToolMessage 对（支持累积多轮）。"""
    if not client_tool_results:
        return lc_messages

    for ctr in client_tool_results:
        args: dict = {"user_request": user_content.strip()}
        try:
            import json

            parsed = json.loads(ctr.result)
            if isinstance(parsed, dict) and parsed.get("user_request"):
                args["user_request"] = str(parsed["user_request"])
        except Exception:
            pass
        lc_messages.append(
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "id": ctr.tool_call_id,
                        "name": ctr.name,
                        "args": args,
                    },
                ],
            ),
        )
        lc_messages.append(
            ToolMessage(content=ctr.result, tool_call_id=ctr.tool_call_id),
        )
    return lc_messages
