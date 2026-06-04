"""Backend-driven ReAct loop for single-image bbox sub-agent; local tools run on the client."""
from __future__ import annotations

import json
import uuid
from collections.abc import AsyncIterator
from typing import Any

from langchain_openai import ChatOpenAI

from app.agent.annotation.agent_turn_service import MessageItem, ToolCallOut, run_agent_turn
from app.agent.annotation.debug_log import log_annotation_agent
from app.agent.annotation.map_labels_service import map_detection_boxes_to_labels_unified
from app.agent.annotation.schemas import AnnotationScopePayload
from app.agent.annotation.sub_image_run_session import (
    SubImageRunSession,
    create_session,
    is_client_tool,
    remove_session,
)
from app.core.config import Settings
from app.services.llm_provider_service import LlmProviderService


def _sse(event_type: str, data: dict[str, Any]) -> dict[str, Any]:
    return {"type": event_type, "data": data}


def _build_initial_human(
    *,
    user_request: str,
    plan: dict[str, Any],
    image_relative_path: str,
    image_absolute_path: str,
    label_candidates: list[dict[str, Any]],
    detection_model_id: str,
) -> str:
    brief = {
        "intent_summary": plan.get("intent_summary"),
        "label_strategy": plan.get("label_strategy"),
        "use_vision_mapping": plan.get("use_vision_mapping"),
        "detection_hints": plan.get("detection_hints"),
        "annotation_scope": plan.get("annotation_scope"),
        "sub_agent_constraints": plan.get("sub_agent_constraints"),
        "detection_model_id": detection_model_id,
    }
    return json.dumps(
        {
            "question": f"请为图片 {image_relative_path} 生成可应用的矩形框标注。",
            "user_request": user_request,
            "brief": brief,
            "label_candidates": label_candidates,
            "file_path": image_absolute_path,
        },
        ensure_ascii=False,
    )


async def _execute_map_on_server(
    *,
    llm: ChatOpenAI | None,
    use_vision: bool,
    user_request: str,
    plan: dict[str, Any],
    label_candidates: list[dict[str, Any]],
    state: Any,
    image_absolute_path: str = "",
    image_base64: str = "",
) -> str:
    boxes = state.boxes
    if not boxes:
        return json.dumps({"ok": False, "error": "无检测框；请先 run_object_detection"}, ensure_ascii=False)

    scope = AnnotationScopePayload.model_validate(plan.get("annotation_scope") or {})
    label_strategy = str(plan.get("label_strategy") or "map_each_box_to_label")
    single_label_id = None
    if label_strategy == "single_label_for_all_boxes" and len(label_candidates) == 1:
        single_label_id = str(label_candidates[0].get("id") or "") or None

    result = await map_detection_boxes_to_labels_unified(
        llm,
        user_request=user_request,
        intent_summary=str(plan.get("intent_summary") or ""),
        label_candidates=label_candidates,
        boxes=boxes,
        use_vision=use_vision,
        scope=scope,
        label_strategy=label_strategy,
        single_label_id=single_label_id,
        image_absolute_path=image_absolute_path,
        image_base64=image_base64,
    )
    state.mappings = list(result.get("mappings") or [])
    state.map_method = str(result.get("method") or "")
    state.map_hint = str(result.get("hint") or "")
    mapped = sum(1 for m in state.mappings if m.get("label_id"))
    unmapped = max(0, len(boxes) - mapped)
    log_annotation_agent(
        "sub-image-map",
        "服务端 map",
        method=state.map_method,
        mapped=mapped,
        unmapped=unmapped,
    )
    min_labeled = int((plan.get("sub_agent_constraints") or {}).get("min_labeled_box_count") or 1)
    auto_ok = mapped >= min_labeled
    if auto_ok:
        state.captured_finalize = True
    return json.dumps(
        {
            "ok": result.get("ok") is not False,
            "method": state.map_method,
            "mapped_count": mapped,
            "unmapped_count": unmapped,
            "hint": state.map_hint,
            "auto_finalized": auto_ok,
            "next_step": result.get("next_step") or "finalize_image_change",
        },
        ensure_ascii=False,
    )


async def stream_sub_image_run(
    *,
    session: SubImageRunSession,
    llm: ChatOpenAI,
    provider_is_vision: bool,
    settings: Settings,
    user_request: str,
    plan: dict[str, Any],
    image_relative_path: str,
    image_absolute_path: str,
    label_candidates: list[dict[str, Any]],
    detection_model_id: str,
    image_base64: str,
) -> AsyncIterator[dict[str, Any]]:
    max_rounds = max(1, int(settings.annotation_sub_agent_max_iterations))
    use_vision = bool(plan.get("use_vision_mapping")) and provider_is_vision
    llm_map = llm if use_vision else None

    messages: list[MessageItem] = [
        MessageItem(role="human", content=_build_initial_human(
            user_request=user_request,
            plan=plan,
            image_relative_path=image_relative_path,
            image_absolute_path=image_absolute_path,
            label_candidates=label_candidates,
            detection_model_id=detection_model_id,
        ))
    ]

    yield _sse(
        "session",
        {
            "run_id": session.run_id,
            "image_relative_path": image_relative_path,
            "max_rounds": max_rounds,
            "use_vision_mapping": use_vision,
        },
    )

    last_content = ""
    try:
        for round_idx in range(max_rounds):
            turn = await run_agent_turn(llm, kind="image", messages=messages)
            last_content = turn.content or ""
            if not turn.tool_calls:
                log_annotation_agent(
                    "sub-image-round",
                    "无 tool_calls",
                    round=round_idx,
                    content_preview=last_content[:200],
                )
                break

            tool_names = [t.name for t in turn.tool_calls]
            yield _sse("round", {"round": round_idx, "tools": tool_names})
            log_annotation_agent("sub-image-round", "tool_calls", round=round_idx, tools=tool_names)

            normalized = []
            for idx, call in enumerate(turn.tool_calls):
                call_id = (call.id or "").strip() or f"sub-{round_idx}-{call.name}-{idx}"
                normalized.append(
                    ToolCallOut(id=call_id, name=call.name, args=call.args or {}),
                )

            messages.append(
                MessageItem(
                    role="assistant",
                    content=turn.content or "",
                    tool_calls=normalized,
                )
            )

            for call in normalized:
                if call.name == "map_detection_boxes_to_labels":
                    arg_boxes = call.args.get("boxes") if isinstance(call.args, dict) else None
                    if isinstance(arg_boxes, list) and arg_boxes:
                        session.state.boxes = [_normalize_box(b, i) for i, b in enumerate(arg_boxes)]
                        session.state.kept_count = len(session.state.boxes)
                    tool_content = await _execute_map_on_server(
                        llm=llm_map,
                        use_vision=use_vision,
                        user_request=user_request,
                        plan=plan,
                        label_candidates=label_candidates,
                        state=session.state,
                        image_absolute_path=image_absolute_path,
                        image_base64=image_base64,
                    )
                elif is_client_tool(call.name):
                    yield _sse(
                        "client_tool",
                        {
                            "run_id": session.run_id,
                            "tool_call_id": call.id,
                            "name": call.name,
                            "args": call.args,
                            "round": round_idx,
                        },
                    )
                    try:
                        tool_content = await session.wait_client_tool(call.id)
                    except Exception as exc:
                        tool_content = json.dumps(
                            {"ok": False, "error": str(exc)[:300]},
                            ensure_ascii=False,
                        )
                    if call.name == "run_object_detection":
                        _apply_detect_result(session.state, tool_content)
                    elif call.name == "finalize_image_change":
                        try:
                            parsed = json.loads(tool_content)
                            if parsed.get("captured"):
                                session.state.captured_finalize = True
                        except json.JSONDecodeError:
                            pass
                else:
                    tool_content = json.dumps(
                        {"ok": False, "error": f"未知工具: {call.name}"},
                        ensure_ascii=False,
                    )

                messages.append(
                    MessageItem(role="tool", content=tool_content, tool_call_id=call.id)
                )

            if session.state.captured_finalize:
                break

        st = session.state
        mapped_count = sum(1 for m in st.mappings if m.get("label_id"))
        yield _sse(
            "done",
            {
                "run_id": session.run_id,
                "ok": mapped_count > 0 or st.captured_finalize,
                "content": last_content,
                "raw_count": st.raw_count,
                "kept_count": st.kept_count,
                "excluded_count": st.excluded_count,
                "mapped_count": mapped_count,
                "unmapped_count": max(0, st.kept_count - mapped_count),
                "method": st.map_method,
                "map_hint": st.map_hint,
                "boxes": st.boxes,
                "mappings": st.mappings,
                "captured_finalize": st.captured_finalize,
            },
        )
    finally:
        await remove_session(session.run_id)


def _normalize_box(raw: Any, idx: int) -> dict[str, Any]:
    if not isinstance(raw, dict):
        return {"box_index": idx}
    return {
        "box_index": int(raw.get("box_index", idx)),
        "class_name": raw.get("class_name") or raw.get("detection_label") or "",
        "confidence": raw.get("confidence"),
        "x": float(raw.get("x", 0)),
        "y": float(raw.get("y", 0)),
        "width": float(raw.get("width", 0)),
        "height": float(raw.get("height", 0)),
    }


def _apply_detect_result(state: Any, tool_content: str) -> None:
    try:
        parsed = json.loads(tool_content)
    except json.JSONDecodeError:
        return
    if not parsed.get("ok"):
        return
    state.raw_count = int(parsed.get("raw_count") or 0)
    state.kept_count = int(parsed.get("kept_count") or 0)
    state.excluded_count = int(parsed.get("excluded_count") or 0)
    boxes = parsed.get("boxes")
    if isinstance(boxes, list):
        state.boxes = boxes


async def submit_client_tool_result(
    run_id: str,
    user_id: uuid.UUID,
    tool_call_id: str,
    content: str,
) -> bool:
    from app.agent.annotation.sub_image_run_session import get_session

    session = await get_session(run_id, user_id)
    if session is None:
        return False
    return session.submit_tool_result(tool_call_id, content)
