"""
Evaluation API endpoints.

Provides retrieval and system-level evaluation information
for the AI Research Assistant.
"""

from fastapi import APIRouter, Depends

from app.core.dependencies import get_hybrid_retriever
from app.services.retrieval.hybrid_retrieval import HybridRetriever


router = APIRouter(
    prefix="/evaluation",
    tags=["Evaluation"],
)


@router.get("/stats")
def get_evaluation_stats(
    retriever: HybridRetriever = Depends(get_hybrid_retriever),
):
    """
    Return current retrieval/index statistics.

    This endpoint is intentionally lightweight for now.
    Detailed benchmark metrics will be added to the
    evaluation framework later.
    """
    return {
        "retrieval": retriever.get_stats(),
        "metrics": {
            "recall_at_5": None,
            "recall_at_10": None,
            "mrr": None,
            "ndcg": None,
            "citation_accuracy": None,
            "groundedness": None,
        },
        "message": "Evaluation statistics endpoint is operational.",
    }