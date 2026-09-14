
"""
Application Dependency Configuration.

Creates and wires application-level services and repositories.

The dependency layer is responsible for composing the application's
components so that API endpoints do not need to manually construct
repositories or services.
"""

from app.repositories.document_repository import DocumentRepository
from app.repositories.vector_repository import VectorRepository
from app.repositories.bm25_repository import BM25Repository

from app.services.embeddings.embedding_service import EmbeddingService
from app.services.retrieval.hybrid_retrieval import HybridRetriever
from app.services.reranker import RerankerService
from app.services.context_builder import ContextBuilder
from app.services.llm.llm_services import LLMService
from app.services.chunkers.character_chunker import TextChunker
from app.services.document_service import DocumentService
from app.services.conversation_memory import ConversationMemory
from app.services.citation_validator import CitationValidator
from app.services.chat_service import ChatService


# ============================================================
# REPOSITORIES
# ============================================================

document_repository = DocumentRepository()

vector_repository = VectorRepository()

bm25_repository = BM25Repository()


# ============================================================
# CORE SERVICES
# ============================================================

embedding_service = EmbeddingService()

chunker = TextChunker()

reranker = RerankerService()

context_builder = ContextBuilder()

llm_service = LLMService()

conversation_memory = ConversationMemory()

citation_validator = CitationValidator()


# ============================================================
# DOCUMENT SERVICE
# ============================================================

document_service = DocumentService(
    repository=document_repository,
    chunker=chunker,
    embedding_service=embedding_service,
    vector_repository=vector_repository,
    bm25_repository=bm25_repository,
)


# ============================================================
# RETRIEVAL SERVICE
# ============================================================

hybrid_retriever = HybridRetriever(
    vector_repository=vector_repository,
    embedding_service=embedding_service,
    bm25_repository=bm25_repository,
    reranker=reranker,
)


# ============================================================
# CHAT SERVICE
# ============================================================

chat_service = ChatService(
    conversation_memory=conversation_memory,
    hybrid_retriever=hybrid_retriever,
    context_builder=context_builder,
    llm_service=llm_service,
    citation_validator=citation_validator,
)


# ============================================================
# DEPENDENCY GETTERS
# ============================================================

def get_document_service() -> DocumentService:
    """
    Return the application document service.
    """

    return document_service


def get_hybrid_retriever() -> HybridRetriever:
    """
    Return the application hybrid retriever.
    """

    return hybrid_retriever


def get_context_builder() -> ContextBuilder:
    """
    Return the application context builder.
    """

    return context_builder


def get_llm_service() -> LLMService:
    """
    Return the application LLM service.
    """

    return llm_service


def get_conversation_memory() -> ConversationMemory:
    """
    Return the application conversation memory.
    """

    return conversation_memory


def get_citation_validator() -> CitationValidator:
    """
    Return the application citation validator.
    """

    return citation_validator


def get_chat_service() -> ChatService:
    """
    Return the application chat service.
    """

    return chat_service

