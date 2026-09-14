
"""
Search API Endpoints.

Provides endpoints for searching the indexed document collection.

The endpoint layer is responsible only for HTTP handling.
Retrieval logic remains inside the retrieval services.
"""

from fastapi import APIRouter, Depends, status

from app.core.dependencies import get_hybrid_retriever
from app.schemas.search import (
    SearchRequest,
    SearchResponse,
    SearchResult,
)
from app.services.retrieval.hybrid_retrieval import HybridRetriever


router = APIRouter(
    prefix="/search",
    tags=["Search"],
)


# ============================================================
# SEARCH
# ============================================================

@router.post(
    "/",
    response_model=SearchResponse,
    status_code=status.HTTP_200_OK,
)
def search(
    request: SearchRequest,
    hybrid_retriever: HybridRetriever = Depends(
        get_hybrid_retriever
    ),
):
    """
    Search the document collection using hybrid retrieval.

    The retrieval pipeline is handled by HybridRetriever.

    The API does not directly access vector or BM25 repositories.
    """

    retrieved_results = hybrid_retriever.search(
        query=request.query,
        limit=request.limit,
    )

    results: list[SearchResult] = []

    for result in retrieved_results:
        chunk = result.get("chunk")

        if chunk is None:
            continue

        results.append(
            SearchResult(
                chunk_id=str(
                    getattr(chunk, "chunk_id", "")
                ),
                text=getattr(
                    chunk,
                    "text",
                    "",
                ),
                source=getattr(
                    chunk,
                    "source",
                    None,
                ),
                page_number=getattr(
                    chunk,
                    "page_number",
                    None,
                ),
                score=float(
                    result.get("score", 0.0)
                ),
            )
        )

    return SearchResponse(
        query=request.query,
        results=results,
        total_results=len(results),
    )


# ============================================================
# SEARCH STATISTICS
# ============================================================

@router.get(
    "/stats",
    status_code=status.HTTP_200_OK,
)
def search_stats(
    hybrid_retriever: HybridRetriever = Depends(
        get_hybrid_retriever
    ),
):
    """
    Return basic retrieval/index statistics.
    """

    return hybrid_retriever.get_stats()

