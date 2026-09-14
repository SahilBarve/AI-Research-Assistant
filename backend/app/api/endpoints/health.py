import requests
from fastapi import APIRouter, Depends, status
from sqlalchemy.orm import Session
from sqlalchemy import text

from app.core.config import settings
from app.database.session import get_db
from app.core.dependencies import vector_repository

# Initialize APIRouter for System Health and Status Monitoring
router = APIRouter(prefix="/health", tags=["Health & System"])


@router.get("/", status_code=status.HTTP_200_OK)
def system_health_check(db: Session = Depends(get_db)):
    """
    REST Endpoint: GET /api/v1/health/

    Performs live connectivity checks across all platform dependencies:
    - PostgreSQL Relational Database
    - Qdrant Vector Database
    - Ollama Local LLM Engine
    """
    # Check PostgreSQL database connection
    postgres_ok = False
    try:
        db.execute(text("SELECT 1"))
        postgres_ok = True
    except Exception:
        pass

    # Check Qdrant collection availability
    qdrant_ok = vector_repository.collection_exists()

    # Check Ollama local service HTTP port
    ollama_ok = False
    try:
        resp = requests.get("http://localhost:11434/api/tags", timeout=2)
        if resp.status_code == 200:
            ollama_ok = True
    except Exception:
        pass

    # Aggregate overall status
    is_healthy = postgres_ok and qdrant_ok and ollama_ok

    return {
        "status": "healthy" if is_healthy else "degraded",
        "components": {
            "postgres": "online" if postgres_ok else "offline",
            "qdrant": "online" if qdrant_ok else "offline",
            "ollama": "online" if ollama_ok else "offline",
        },
    }


@router.get("/system/stats", status_code=status.HTTP_200_OK)
def get_system_stats():
    """
    REST Endpoint: GET /api/v1/health/system/stats

    Returns system runtime configuration, active AI model metadata,
    and total indexed vector chunk counts.
    """
    # Fetch all stored points via paginated scroll to get actual vector count
    points = vector_repository.get_all_points_paginated(batch_size=100)

    return {
        "app_name": settings.APP_NAME,
        "version": settings.APP_VERSION,
        "embedding_model": settings.EMBEDDING_MODEL,
        "llm_model": settings.LLM_MODEL,
        "total_vector_chunks": len(points),
    }