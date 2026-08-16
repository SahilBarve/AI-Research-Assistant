"""
Application service dependencies.

Creates and stores shared instances of the services
used by the RAG pipeline.
"""

from app.repositories.vector_repository import VectorRepository
from app.repositories.bm25_repository import BM25Repository
from app.services.embeddings.embedding_service import EmbeddingService
from app.services.retrieval.hybrid_retrieval import HybridRetriever
from app.services.reranker import RerankerService
from app.services.context_builder import ContextBuilder
from app.services.llm.llm_services import LLMService
from app.schemas.chunk import DocumentChunk


# =========================================================
# SHARED SERVICES
# =========================================================

vector_repository = VectorRepository()

embedding_service = EmbeddingService()

bm25_repository = BM25Repository()

reranker = RerankerService()

context_builder = ContextBuilder()

llm_service = LLMService()


# =========================================================
# BUILD BM25 INDEX
# =========================================================

stored_points = vector_repository.get_all_points(
    limit=1000
)

chunks = []

for point in stored_points:

    payload = point.payload

    chunks.append(
        DocumentChunk(
            chunk_id=payload["chunk_id"],
            text=payload["text"],
            source=payload["source"],
        )
    )


if chunks:
    bm25_repository.build_index(chunks)


# =========================================================
# HYBRID RETRIEVER
# =========================================================

hybrid_retriever = HybridRetriever(
    vector_repository=vector_repository,
    embedding_service=embedding_service,
    bm25_repository=bm25_repository,
    reranker=reranker,
)


# =========================================================
# DEPENDENCY GETTERS
# =========================================================

def get_hybrid_retriever():
    return hybrid_retriever


def get_context_builder():
    return context_builder


def get_llm_service():
    return llm_service