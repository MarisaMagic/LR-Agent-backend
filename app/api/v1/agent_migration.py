"""Data migration endpoint — exports all user sessions/messages for local storage migration."""

import json
import logging
from fastapi import APIRouter, HTTPException, status
from sqlalchemy import select, text

from app.core.deps import CurrentUser, DbSession, SettingsDep
from app.models.agent_session import AgentSession
from app.models.agent_message import AgentMessage

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/agent/export", tags=["agent-export"])


@router.get("/sessions")
async def export_sessions(
    current_user: CurrentUser,
    db: DbSession,
) -> dict:
    """Export all sessions and messages for the current user.
    
    Returns complete session and message data in a format compatible
    with the frontend local SQLite schema.
    """
    # Get all non-deleted sessions
    sessions_result = await db.execute(
        select(AgentSession).where(
            AgentSession.user_id == current_user.id,
            AgentSession.deleted_at.is_(None),
        ).order_by(AgentSession.updated_at.desc())
    )
    sessions = sessions_result.scalars().all()

    exported_sessions = []
    for session in sessions:
        # Get all messages for this session
        messages_result = await db.execute(
            select(AgentMessage).where(
                AgentMessage.session_id == session.id,
            ).order_by(AgentMessage.sort_index.asc())
        )
        messages = messages_result.scalars().all()

        exported_sessions.append({
            "id": session.id,
            "title": session.title,
            "annotation_project_id": session.annotation_project_id,
            "interaction_mode": session.interaction_mode,
            "provider_id": session.provider_id,
            "model": session.model,
            "context_summary": session.context_summary,
            "summary_up_to_message_id": session.summary_up_to_message_id,
            "last_context_token_estimate": session.last_context_token_estimate,
            "created_at": int(session.created_at.timestamp() * 1000) if session.created_at else 0,
            "updated_at": int(session.updated_at.timestamp() * 1000) if session.updated_at else 0,
            "messages": [
                {
                    "id": msg.id,
                    "session_id": msg.session_id,
                    "role": msg.role,
                    "interaction_mode": msg.interaction_mode,
                    "sort_index": msg.sort_index,
                    "blocks_json": json.dumps(msg.blocks_json, ensure_ascii=False) if isinstance(msg.blocks_json, list) else str(msg.blocks_json),
                    "status": msg.status,
                    "provider_id": msg.provider_id,
                    "model": msg.model,
                    "error": msg.error,
                    "created_at": int(msg.created_at.timestamp() * 1000) if msg.created_at else 0,
                    "updated_at": int(msg.updated_at.timestamp() * 1000) if msg.updated_at else 0,
                }
                for msg in messages
            ],
        })

    return {"sessions": exported_sessions, "total": len(exported_sessions)}


@router.delete("/clear")
async def clear_exported_data(
    current_user: CurrentUser,
    db: DbSession,
) -> dict:
    """Clear all sessions and messages for the current user after migration.
    
    This permanently deletes all chat data. Only call after confirming
    the data has been successfully migrated to the local SQLite database.
    """
    # Delete all messages for user's sessions
    messages_deleted = await db.execute(
        text("""
            DELETE FROM agent_messages 
            WHERE user_id = :user_id
        """),
        {"user_id": current_user.id}
    )

    # Delete all sessions
    sessions_deleted = await db.execute(
        text("""
            DELETE FROM agent_sessions 
            WHERE user_id = :user_id
        """),
        {"user_id": current_user.id}
    )

    # Delete all LLM provider configs
    providers_deleted = await db.execute(
        text("""
            DELETE FROM llm_providers 
            WHERE user_id = :user_id
        """),
        {"user_id": current_user.id}
    )

    await db.commit()

    logger.info(
        "Data cleared for user %s: sessions=%d messages=%d providers=%d",
        current_user.id,
        sessions_deleted.rowcount,
        messages_deleted.rowcount,
        providers_deleted.rowcount,
    )

    return {
        "status": "cleared",
        "sessions_deleted": sessions_deleted.rowcount,
        "messages_deleted": messages_deleted.rowcount,
        "providers_deleted": providers_deleted.rowcount,
    }
