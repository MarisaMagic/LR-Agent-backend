from fastapi import APIRouter

from app.api.v1 import (
    agent,
    agent_analysis,
    annotation_agent,
    annotation_quality,
    auth,
    users,
)

api_router = APIRouter()
api_router.include_router(auth.router)
api_router.include_router(users.router)
api_router.include_router(agent.router)
api_router.include_router(annotation_agent.router)
api_router.include_router(agent_analysis.router)
api_router.include_router(annotation_quality.router)
