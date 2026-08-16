from fastapi import APIRouter

from app.schemas.chat import ChatRequest, ChatResponse
from app.core.dependencies import (
    get_hybrid_retriever,
    get_context_builder,
    get_llm_service,
)


router = APIRouter(
    prefix="/chat",
    tags=["Chat"],
)


@router.post(
    "/",
    response_model=ChatResponse,
)
def chat(request: ChatRequest):

    # =====================================================
    # STEP 1: GET SERVICES
    # =====================================================

    hybrid_retriever = get_hybrid_retriever()
    context_builder = get_context_builder()
    llm_service = get_llm_service()

    # =====================================================
    # STEP 2: RETRIEVE RELEVANT CHUNKS
    # =====================================================

    retrieved_results = hybrid_retriever.search(
        query=request.question,
        limit=5,
        retrieval_limit=30,
        rerank_limit=20,
    )

    # =====================================================
    # STEP 3: BUILD LLM CONTEXT
    # =====================================================

    context = context_builder.build_context(
        retrieved_results
    )

    # =====================================================
    # STEP 4: GENERATE ANSWER
    # =====================================================

    answer = llm_service.generate(
        query=request.question,
        context=context,
    )

    # =====================================================
    # STEP 5: RETURN RESPONSE
    # =====================================================

    return ChatResponse(
        answer=answer
    )