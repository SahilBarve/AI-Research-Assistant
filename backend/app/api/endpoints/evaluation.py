"""
Evaluation API endpoints.

Provides retrieval and system-level evaluation information
for the AI Research Assistant.
"""

from fastapi import APIRouter, Depends

from app.core.dependencies import get_hybrid_retriever
from app.services.evaluation_service import EvaluationService
from app.services.retrieval.hybrid_retrieval import HybridRetriever


router = APIRouter(
    prefix="/evaluation",
    tags=["Evaluation"],
)


# ============================================================
# EVALUATION
# ============================================================

@router.get("/run")
def run_evaluation(
    retriever: HybridRetriever = Depends(
        get_hybrid_retriever
    ),
):
    """
    Run the retrieval evaluation benchmark.

    Evaluates the current hybrid + reranker pipeline
    against the manually curated evaluation dataset.
    """

    evaluation_service = EvaluationService(
        retriever= retriever,
    )

    return evaluation_service.evaluate()


# ============================================================
# RETRIEVAL STATISTICS
# ============================================================

@router.get("/stats")
def get_evaluation_stats(
    retriever: HybridRetriever = Depends(
        get_hybrid_retriever
    ),
):
    """
    Return current retrieval/index statistics.
    """

    return {
        "retrieval": retriever.get_stats(),
        "message": "Evaluation statistics endpoint is operational.",
    }