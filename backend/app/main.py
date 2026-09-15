
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI

from app.api.router import api_router
from app.core.config import get_settings
from app.core.logging import logger
from app.core.dependencies import vector_repository, bm25_repository
from app.exceptions.handlers import register_exception_handlers
from app.schemas.chunk import DocumentChunk


settings = get_settings()


def rebuild_bm25_index() -> None:
    """
    Rebuild the in-memory BM25 index from the durable Qdrant store.

    Qdrant is the source of truth for indexed document chunks.
    BM25 is kept in memory for fast lexical retrieval and rebuilt
    whenever the application starts.
    """

    try:
        points = vector_repository.get_all_points_paginated(
            batch_size=100
        )

        chunks = []

        for point in points:
            payload = point.payload or {}

            # Skip malformed points instead of crashing
            # the entire application startup.
            if not payload.get("text") or not payload.get("source"):
                continue

            chunks.append(
                DocumentChunk(
                    chunk_id=int(payload["chunk_id"]),
                    text=str(payload["text"]),
                    source=str(payload["source"]),
                    page_number=int(payload.get("page_number", 1)),
                )
            )

        bm25_repository.rebuild_from_chunks(chunks)

        logger.info(
            "BM25 index rebuilt successfully. "
            "Indexed chunks: %d",
            len(chunks),
        )

    except Exception:
        logger.exception(
            "Failed to rebuild BM25 index from Qdrant."
        )


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Application startup and shutdown lifecycle.

    Startup:
        - Ensure upload directory exists.
        - Rebuild BM25 from Qdrant.

    Shutdown:
        - Log application shutdown.
    """

    # ---------------------------------------------------------
    # STARTUP
    # ---------------------------------------------------------

    Path(settings.upload_dir).mkdir(
        parents=True,
        exist_ok=True,
    )

    logger.info(
        "Starting AI Research Assistant..."
    )

    rebuild_bm25_index()

    logger.info(
        "AI Research Assistant startup completed."
    )

    yield

    # ---------------------------------------------------------
    # SHUTDOWN
    # ---------------------------------------------------------

    logger.info(
        "AI Research Assistant shutting down."
    )


app = FastAPI(
    title="AI Research Assistant",
    version="1.0.0",
    description="Production-grade AI Knowledge Intelligence Platform",
    lifespan=lifespan,
)

register_exception_handlers(app)

app.include_router(api_router)


@app.get("/")
def root():
    return {
        "status": "running",
        "message": "Welcome to AI Research Assistant 🚀",
    }

