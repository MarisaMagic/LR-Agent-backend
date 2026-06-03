from fastapi import APIRouter

from app.api.v1 import agent, auth, llm_providers, users

api_router = APIRouter()
api_router.include_router(auth.router)
api_router.include_router(users.router)
api_router.include_router(llm_providers.router)
api_router.include_router(agent.router)
