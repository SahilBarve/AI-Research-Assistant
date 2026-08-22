"""
Application service dependencies.

Creates and stores shared instances of the services
used by the RAG pipeline.
"""

from app.repositories.vector_repository import VectorRepository
from app.repositories.bm25_repository import BM25Repository
from app.repositories.document_repository import DocumentRepository

from app.services.embeddings.embedding_service import EmbeddingService
from app.services.retrieval.hybrid_retrieval import HybridRetriever
from app.services.reranker import RerankerService
from app.services.context_builder import ContextBuilder
from app.services.llm.llm_services import LLMService

from app.services.document_service import DocumentService
from app.services.document_processors.pdf_processors import PDFProcessor
from app.services.chunkers.character_chunker import TextChunker

from app.schemas.chunk import DocumentChunk
from app.services.conversation_memory import ConversationMemory

# =========================================================
# SHARED REPOSITORIES
# =========================================================

vector_repository = VectorRepository()

bm25_repository = BM25Repository()

document_repository = DocumentRepository()


# =========================================================
# SHARED SERVICES
# =========================================================

embedding_service = EmbeddingService()

reranker = RerankerService()

context_builder = ContextBuilder()

llm_service = LLMService()

conversation_memory = ConversationMemory()


# =========================================================
# DOCUMENT PROCESSING SERVICES
# =========================================================

pdf_processor = PDFProcessor()

text_chunker = TextChunker()


# =========================================================
# INITIALIZE BM25 FROM QDRANT
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
            page_number=payload["page_number"],
        )
    )


if chunks:

    bm25_repository.build_index(
        chunks
    )


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
# DOCUMENT SERVICE
# =========================================================

document_service = DocumentService(
    repository=document_repository,
    processor=pdf_processor,
    chunker=text_chunker,
    embedding_service=embedding_service,
    vector_repository=vector_repository,
    bm25_repository=bm25_repository,
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


def get_document_service():
    return document_service

def get_conversation_memory():
    return conversation_memory