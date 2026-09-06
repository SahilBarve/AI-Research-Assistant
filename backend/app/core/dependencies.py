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

# Global Singletons
document_repository = DocumentRepository()
vector_repository = VectorRepository()
bm25_repository = BM25Repository()

embedding_service = EmbeddingService()
chunker = TextChunker()
reranker = RerankerService()
context_builder = ContextBuilder()
llm_service = LLMService()
conversation_memory = ConversationMemory()

document_service = DocumentService(
    repository=document_repository,
    chunker=chunker,
    embedding_service=embedding_service,
    vector_repository=vector_repository,
    bm25_repository=bm25_repository,
)

hybrid_retriever = HybridRetriever(
    vector_repository=vector_repository,
    embedding_service=embedding_service,
    bm25_repository=bm25_repository,
    reranker=reranker,
)

def get_document_service() -> DocumentService:
    return document_service

def get_hybrid_retriever() -> HybridRetriever:
    return hybrid_retriever

def get_context_builder() -> ContextBuilder:
    return context_builder

def get_llm_service() -> LLMService:
    return llm_service

def get_conversation_memory() -> ConversationMemory:
    return conversation_memory