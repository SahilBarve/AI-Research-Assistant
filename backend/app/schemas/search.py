
"""
Search API Schemas.

Defines request and response models for the retrieval/search API.
"""

from pydantic import BaseModel, Field


# ============================================================
# SEARCH REQUEST
# ============================================================

class SearchRequest(BaseModel):
    """
    Request model for semantic/hybrid search.
    """

    query: str = Field(
        ...,
        min_length=1,
        max_length=1000,
        description="Search query.",
    )

    limit: int = Field(
        default=5,
        ge=1,
        le=50,
        description="Maximum number of results to return.",
    )


# ============================================================
# SEARCH RESULT
# ============================================================

class SearchResult(BaseModel):
    """
    Represents a single retrieved document chunk.
    """

    chunk_id: str

    text: str

    source: str | None = None

    page_number: int | None = None

    score: float


# ============================================================
# SEARCH RESPONSE
# ============================================================

class SearchResponse(BaseModel):
    """
    Response returned by the search endpoint.
    """

    query: str

    results: list[SearchResult]

    total_results: int

