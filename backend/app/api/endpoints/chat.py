from fastapi import APIRouter

from app.schemas.chat import (
    ChatRequest,
    ChatResponse,
)

from app.core.dependencies import (
    get_hybrid_retriever,
    get_context_builder,
    get_llm_service,
    get_conversation_memory,
)

from app.services.citation_validator import (
    CitationValidator,
)


# =========================================================
# ROUTER
# =========================================================

router = APIRouter(
    prefix="/chat",
    tags=["Chat"],
)


# =========================================================
# CITATION VALIDATOR
# =========================================================

citation_validator = CitationValidator()


# =========================================================
# CHAT ENDPOINT
# =========================================================

@router.post(
    "/",
    response_model=ChatResponse,
)
def chat(
    request: ChatRequest,
):

    # =====================================================
    # STEP 1
    # GET SHARED SERVICES
    # =====================================================

    hybrid_retriever = (
        get_hybrid_retriever()
    )

    context_builder = (
        get_context_builder()
    )

    llm_service = (
        get_llm_service()
    )

    conversation_memory = (
        get_conversation_memory()
    )

    # =====================================================
    # STEP 2
    # GET PREVIOUS CONVERSATION
    # =====================================================

    conversation_history = (
        conversation_memory.format_history(
            session_id=request.session_id,
            max_messages=10,
        )
    )

    # =====================================================
    # STEP 3
    # STORE USER MESSAGE
    # =====================================================

    conversation_memory.add_user_message(
        session_id=request.session_id,
        message=request.question,
    )

    # =====================================================
    # STEP 4
    # HYBRID RETRIEVAL
    # =====================================================

    retrieved_results = (
        hybrid_retriever.search(
            query=request.question,
            limit=5,
            retrieval_limit=30,
            rerank_limit=20,
        )
    )

    # =====================================================
    # STEP 5
    # BUILD CONTEXT
    # =====================================================

    context_data = (
        context_builder.build_context(
            retrieved_results
        )
    )

    context = context_data["context"]

    citations = context_data["citations"]

    # =====================================================
    # STEP 6
    # GENERATE ANSWER
    # =====================================================

    answer = llm_service.generate(
        query=request.question,
        context=context,
        conversation_history=conversation_history,
    )

    # =====================================================
    # STEP 7
    # VALIDATE CITATIONS
    # =====================================================

    valid_citations, invalid_citations = (
        citation_validator.validate(
            answer=answer,
            citations=citations,
        )
    )

    # =====================================================
    # STEP 8
    # REMOVE INVALID CITATIONS
    # =====================================================

    if invalid_citations:

        print(
            "\nWARNING: Invalid citation IDs:",
            invalid_citations,
        )

        answer = (
            citation_validator.remove_invalid_citations(
                answer=answer,
                citations=citations,
            )
        )

    # =====================================================
    # STEP 9
    # SAVE ASSISTANT RESPONSE
    # =====================================================

    conversation_memory.add_assistant_message(
        session_id=request.session_id,
        message=answer,
    )

    # =====================================================
    # STEP 10
    # LOG VALIDATION
    # =====================================================

    print(
        "\nValid citations:",
        [
            citation["id"]
            for citation in valid_citations
        ],
    )

    print(
        "Invalid citations:",
        invalid_citations,
    )

    # =====================================================
    # STEP 11
    # RETURN RESPONSE
    # =====================================================

    return ChatResponse(
        answer=answer,
        citations=valid_citations,
    )