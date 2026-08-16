from fastapi import APIRouter

from app.schemas.chat import ChatRequest, ChatResponse
from app.services.retrieval.hybrid_retrieval import HybridRetriever
from app.services.context_builder import ContextBuilder
from app.services.llm.llm_services import LLMService

from app.repositories.vector_repository import VectorRepository
from app.repositories.bm25_repository import BM25Repository
from app.services.embeddings.embedding_service import EmbeddingService
from app.services.reranker import RerankerService


router = APIRouter(
    prefix="/chat",
    tags=["Chat"],
)


# =========================================================
# SERVICE INITIALIZATION
# =========================================================

embedding_service = EmbeddingService()

vector_repository = VectorRepository()

bm25_repository = BM25Repository()

reranker = RerankerService()

hybrid_retriever = HybridRetriever(
    vector_repository=vector_repository,
    embedding_service=embedding_service,
    bm25_repository=bm25_repository,
    reranker=reranker,
)

context_builder = ContextBuilder()

llm_service = LLMService()


# =========================================================
# CHAT ENDPOINT
# =========================================================

@router.post(
    "/",
    response_model=ChatResponse,
)
def chat(request: ChatRequest):

    # -----------------------------------------------------
    # STEP 1: Retrieve relevant chunks
    # -----------------------------------------------------

    retrieved_results = hybrid_retriever.search(
        query=request.question,
        limit=5,
        retrieval_limit=30,
        rerank_limit=20,
    )

    # -----------------------------------------------------
    # STEP 2: Extract DocumentChunks
    # -----------------------------------------------------

    chunks = [
        result["chunk"]
        for result in retrieved_results
    ]

    # -----------------------------------------------------
    # STEP 3: Build LLM context
    # -----------------------------------------------------

    context = context_builder.build_context(
        chunks
    )

    # -----------------------------------------------------
    # STEP 4: Generate answer
    # -----------------------------------------------------

    answer = llm_service.generate(
        query=request.question,
        context=context,
    )

    # -----------------------------------------------------
    # STEP 5: Return response
    # -----------------------------------------------------

    return ChatResponse(
        answer=answer
    )