from fastapi import APIRouter
from app.api.endpoints import health, documents, chat, search, evaluation

# Initialize master API router
api_router = APIRouter()

# Register sub-routers for each feature module
api_router.include_router(health.router)
api_router.include_router(documents.router)
api_router.include_router(chat.router)
api_router.include_router(search.router)
api_router.include_router(evaluation.router)